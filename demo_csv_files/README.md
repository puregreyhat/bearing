# Website test signals

Each CSV contains 4,096 real CWRU Drive-End acceleration samples at 12 kHz (two 2,048-sample analysis windows).

Upload any CSV through **Load 12 kHz signal CSV** on the website. The folder name is the expected result:

- `healthy/097_normal.csv` → Healthy / Normal
- `faulty/inner_race/105_inner_race.csv` → Faulty / Inner Race
- `faulty/ball/118_ball.csv` → Faulty / Ball
- `faulty/outer_race/130_outer_race.csv` → Faulty / Outer Race

These are demonstration signals from the training benchmark, not independent field-validation data.
