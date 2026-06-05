from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

sys.stdout.reconfigure(encoding='utf-8')

try:
    from PyPDF2 import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None  # type: ignore

TODO_PATTERN = re.compile(r"\[TODO(?::\s*([^]]*))?\]|\bTODO\b[^\]\n\r]*", flags=re.IGNORECASE)


def extract_todos_from_pdf(pdf_path: Path) -> list[tuple[int, str]]:
    if PdfReader is None:
        raise RuntimeError("PyPDF2 is required to parse PDF files. Install it with `pip install PyPDF2`.")

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(pdf_path.open('rb'))
    todos: list[tuple[int, str]] = []
    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ''
        for match in TODO_PATTERN.finditer(text):
            start = max(0, match.start() - 40)
            end = min(len(text), match.end() + 40)
            context = text[start:end].replace('\n', ' ').replace('\r', ' ')
            todo_text = match.group(0).strip()
            line_clean = ' '.join(context.split())
            todos.append((page_index + 1, todo_text + ' | context: ' + line_clean))
    return todos


def extract_todos_from_text(text_path: Path) -> list[str]:
    if not text_path.exists():
        return []
    text = text_path.read_text(encoding='utf-8', errors='ignore')
    todos = []
    for line in text.splitlines():
        if 'TODO' in line:
            todos.append(line.strip())
    return todos


def parse_workflow_status(status_path: Path) -> list[tuple[str, str, str]]:
    if not status_path.exists():
        return []

    tasks: list[tuple[str, str, str]] = []
    text = status_path.read_text(encoding='utf-8', errors='ignore')
    for line in text.splitlines():
        if not line.startswith('|'):
            continue
        if 'Description' in line and 'Status' in line:
            continue
        if '---' in line:
            continue

        parts = [cell.strip() for cell in line.split('|')[1:-1]]
        if len(parts) >= 4:
            phase, description, status, details = parts[:4]
            status_clean = re.sub(r'[\*_]', '', status).strip().upper()
            if status_clean != 'DONE':
                tasks.append((phase, status, details))
    return tasks


def format_todo_report(pdf_tasks: list[tuple[int, str]], workflow_tasks: list[tuple[str, str, str]], extra_todos: Iterable[str]) -> str:
    lines: list[str] = []
    lines.append('# Task Checker Report')
    lines.append('')
    lines.append('This report is generated from `HCI.pdf` TODO markers and the workflow status file.')
    lines.append('')
    if pdf_tasks:
        lines.append('## TODO items found in HCI.pdf')
        lines.append('')
        for page, todo in pdf_tasks:
            lines.append(f'- Page {page}: {todo}')
        lines.append('')
    else:
        lines.append('## No TODO markers found in HCI.pdf')
        lines.append('')

    if workflow_tasks:
        lines.append('## Incomplete workflow phases from workflow_status.md')
        lines.append('')
        for phase, status, details in workflow_tasks:
            lines.append(f'- {phase} — {status}: {details}')
        lines.append('')

    if extra_todos:
        lines.append('## Additional TODO lines found in text files')
        lines.append('')
        for todo in extra_todos:
            lines.append(f'- {todo}')
        lines.append('')

    lines.append('## How to use this task checker')
    lines.append('')
    lines.append('1. Run `python task_checker.py` to print undone tasks to the console.')
    lines.append('2. Use `python task_checker.py --output task_check_report.md` to write a Markdown report.')
    lines.append('3. Update the PDF or workflow files, then rerun the checker to confirm the remaining undone items.')
    lines.append('')
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description='Scan HCI.pdf and workflow documents for undone work items.')
    parser.add_argument('--pdf', type=Path, default=Path('HCI.pdf'), help='Path to the HCI PDF file.')
    parser.add_argument('--workflow', type=Path, default=Path('workflow_status.md'), help='Path to the workflow status Markdown file.')
    parser.add_argument('--output', type=Path, help='Optional output Markdown file path.')
    args = parser.parse_args()

    pdf_tasks = extract_todos_from_pdf(args.pdf)
    workflow_tasks = parse_workflow_status(args.workflow)

    extra_todos = []
    for path in [Path('research_workflow.md'), Path('README.md')]:
        extra_todos.extend(extract_todos_from_text(path))

    report = format_todo_report(pdf_tasks, workflow_tasks, extra_todos)

    if args.output:
        args.output.write_text(report, encoding='utf-8')
        print(f'Wrote task report to {args.output}')
    else:
        print(report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
