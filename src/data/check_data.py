import sys
from data import load_csv, validate, resample

path = sys.argv[1]                                   # CSV path from the command line
tz = sys.argv[2] if len(sys.argv) > 2 else "auto"    # optional: UTC, +2, Europe/Athens...

df = load_csv(path, source_tz=tz)
print("loaded:", len(df), "rows |", df.index[0], "->", df.index[-1])

clean, report = validate(df)
print(report.summary())
print(report.gaps[report.gaps["kind"] == "unexpected_long"])   # holes worth inspecting

print(resample(clean, "M15").tail())