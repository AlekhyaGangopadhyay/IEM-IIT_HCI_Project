| File     | Expected   | Model   | Normalization   | Accuracy (%)   | Dominant Predict   | Dominant Pct (%)   | Shifting Rate (%)   |
|:---------|:-----------|:--------|:----------------|:---------------|:-------------------|:-------------------|:--------------------|
| LE.xlsx  | Left       | CNN     | Static          | 11.11%         | Forward            | 77.8%              | 11.1%               |
| LE.xlsx  | Left       | CNN     | Adaptive        | 0.00%          | Backward           | 44.4%              | 33.3%               |
| LE.xlsx  | Left       | LSTM    | Static          | 22.22%         | Forward            | 55.6%              | 22.2%               |
| LE.xlsx  | Left       | LSTM    | Adaptive        | 22.22%         | Forward            | 55.6%              | 11.1%               |
| RY.xlsx  | Right      | CNN     | Static          | 11.11%         | Forward            | 66.7%              | 0.0%                |
| RY.xlsx  | Right      | CNN     | Adaptive        | 22.22%         | Forward            | 55.6%              | 33.3%               |
| RY.xlsx  | Right      | LSTM    | Static          | 11.11%         | Forward            | 55.6%              | 11.1%               |
| RY.xlsx  | Right      | LSTM    | Adaptive        | 11.11%         | Forward            | 55.6%              | 11.1%               |
| For.xlsx | Forward    | CNN     | Static          | 66.67%         | Forward            | 66.7%              | 11.1%               |
| For.xlsx | Forward    | CNN     | Adaptive        | 77.78%         | Forward            | 77.8%              | 44.4%               |
| For.xlsx | Forward    | LSTM    | Static          | 33.33%         | Forward            | 33.3%              | 11.1%               |
| For.xlsx | Forward    | LSTM    | Adaptive        | 44.44%         | Forward            | 44.4%              | 22.2%               |