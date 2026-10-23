"""Tests for the data layer. Run from the xau-setups root:  python -m pytest tests/ -v"""
import numpy as np
import pandas as pd
import pytest

from data import load_csv, read_raw, detect_utc_offset, validate, find_gaps, resample
from data.loader import _parse_source_tz

NY = "America/New_York"


# ------------------------------------------------------------------ helpers
def make_m1(start, end, seed=0, base=2060.0):
    """Synthetic gold M1 bars (UTC index) with the real weekly/daily closures.

    Market is closed Fri 17:00 ET -> Sun 17:00 ET and 17:00-18:00 ET each day,
    which reproduces both winter (23:00 UTC) and summer (22:00 UTC) reopens.
    """
    idx = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left")
    ny = idx.tz_convert(NY)
    wd, h = ny.day_of_week, ny.hour
    closed = ((wd == 4) & (h >= 17)) | (wd == 5) | ((wd == 6) & (h < 17)) | (h == 17)
    idx = idx[~closed]
    rng = np.random.default_rng(seed)
    close = base + np.cumsum(rng.normal(0, 0.2, len(idx)))
    open_ = np.r_[base, close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0, 0.15, len(idx))
    low = np.minimum(open_, close) - rng.uniform(0, 0.15, len(idx))
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                       "volume": rng.integers(1, 120, len(idx))}, index=idx)
    df.index.name = "time"
    return df.round(2)


def write_csv(df, path, offset_hours=0, sep=",", native=False, spread=False):
    """Write df the way MT5 would, in broker time = UTC + offset_hours."""
    t = (df.index + pd.Timedelta(hours=offset_hours)).tz_localize(None)
    if native:
        out = pd.DataFrame({
            "<DATE>": t.strftime("%Y.%m.%d"), "<TIME>": t.strftime("%H:%M:%S"),
            "<OPEN>": df["open"].values, "<HIGH>": df["high"].values,
            "<LOW>": df["low"].values, "<CLOSE>": df["close"].values,
            "<TICKVOL>": df["volume"].values, "<VOL>": 0,
            "<SPREAD>": 20})
        out.to_csv(path, sep="\t", index=False)
        return
    out = pd.DataFrame({"time": t.strftime("%Y-%m-%d %H:%M:%S"), "open": df["open"].values,
                        "high": df["high"].values, "low": df["low"].values,
                        "close": df["close"].values, "volume": df["volume"].values})
    if spread:
        out["spread"] = 18
    out.to_csv(path, sep=sep, index=False)


@pytest.fixture(scope="module")
def winter():
    return make_m1("2024-01-02", "2024-02-16")          # ~6 weeks, no DST change


# ------------------------------------------------------------------ loader
def test_sample_layout_exact(tmp_path):
    """The layout from the user's screenshot."""
    p = tmp_path / "x.csv"
    p.write_text(
        "time,open,high,low,close,volume\n"
        "2024-01-02 05:01:00,2062.14,2069.79,2062.14,2069.79,4\n"
        "2024-01-02 05:02:00,2069.79,2069.79,2069.49,2069.53,66\n")
    df = load_csv(p, source_tz="UTC")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert str(df.index.tz) == "UTC"
    assert df.index[0] == pd.Timestamp("2024-01-02 05:01:00", tz="UTC")
    assert df["close"].iloc[1] == 2069.53 and df["volume"].iloc[0] == 4


def test_native_mt5_export(tmp_path, winter):
    p = tmp_path / "native.csv"
    write_csv(winter.iloc[:50], p, native=True)
    df = load_csv(p, source_tz="UTC")
    assert "spread" in df.columns and (df["spread"] == 20).all()
    assert (df["volume"].values == winter.iloc[:50]["volume"].values).all()   # TICKVOL, not VOL
    assert df.index[0] == winter.index[0]


def test_semicolon_and_uppercase_headers(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("Time;Open;High;Low;Close;Volume\n2024-01-02 05:01:00;1;2;0.5;1.5;7\n")
    df = load_csv(p, source_tz="UTC")
    assert df["high"].iloc[0] == 2 and df["volume"].iloc[0] == 7


def test_missing_column_fails_loudly(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("time,open,high,low,close\n2024-01-02 05:01:00,1,2,0.5,1.5\n")
    with pytest.raises(ValueError, match="volume"):
        load_csv(p, source_tz="UTC")


@pytest.mark.parametrize("offset", [0, 2, 3, -5])
def test_explicit_offset_converts_to_utc(tmp_path, winter, offset):
    p = tmp_path / "o.csv"
    write_csv(winter.iloc[:30], p, offset_hours=offset)
    df = load_csv(p, source_tz=f"{offset:+d}" if offset else "UTC")
    assert (df.index == winter.index[:30]).all()


def test_named_timezone(tmp_path, winter):
    p = tmp_path / "n.csv"
    write_csv(winter.iloc[:30], p, offset_hours=2)          # Athens = UTC+2 in January
    df = load_csv(p, source_tz="Europe/Athens")
    assert (df.index == winter.index[:30]).all()


def test_parse_source_tz_variants():
    assert _parse_source_tz("UTC+2") == pd.Timedelta(hours=2).to_pytimedelta()
    assert _parse_source_tz("-5") == pd.Timedelta(hours=-5).to_pytimedelta()
    with pytest.raises(Exception):
        _parse_source_tz("Not/AZone")


@pytest.mark.parametrize("offset", [0, 2, -5, 3])
def test_auto_detect_recovers_offset(tmp_path, winter, offset):
    p = tmp_path / "a.csv"
    write_csv(winter, p, offset_hours=offset)
    df = load_csv(p, source_tz="auto")
    assert (df.index == winter.index).all()


def test_auto_detect_across_dst_with_fixed_offset(tmp_path):
    """Fixed UTC+2 broker over the US DST change (Mar 10 2024) must still read +2."""
    data = make_m1("2024-02-19", "2024-04-05")
    p = tmp_path / "d.csv"
    write_csv(data, p, offset_hours=2)
    raw = read_raw(p)
    off, agree = detect_utc_offset(raw["time"])
    assert off == 2 and agree == 1.0


def test_auto_detect_refuses_dst_following_broker(tmp_path):
    """Server time that shifts with DST can't be a fixed offset -> must refuse, not guess."""
    a = make_m1("2024-01-08", "2024-03-01")                       # broker at +2
    b = make_m1("2024-04-01", "2024-05-24")                       # broker at +3
    p = tmp_path / "m.csv"
    ta = (a.index + pd.Timedelta(hours=2)).tz_localize(None)
    tb = (b.index + pd.Timedelta(hours=3)).tz_localize(None)
    both = pd.concat([a.set_axis(ta), b.set_axis(tb)])
    both.insert(0, "time", both.index.strftime("%Y-%m-%d %H:%M:%S"))
    both.to_csv(p, index=False)
    with pytest.raises(ValueError, match="inconsistent|explicitly"):
        load_csv(p, source_tz="auto")


def test_auto_detect_refuses_when_too_little_data(tmp_path, winter):
    p = tmp_path / "short.csv"
    write_csv(winter.iloc[:600], p)
    with pytest.raises(ValueError, match="auto-detect"):
        load_csv(p, source_tz="auto")


# ---------------------------------------------------------------- validate
def test_clean_data_passes_untouched(winter):
    clean, rep = validate(winter)
    assert rep.rows_removed == 0 and len(clean) == len(winter)
    assert not rep.unsorted


def test_validate_removes_corrupt_rows_and_reports(winter):
    d = winter.iloc[:1000].copy()
    d.iloc[10, d.columns.get_loc("open")] = np.nan                    # NaN
    d.iloc[20, d.columns.get_loc("low")] = -1.0                       # non-positive
    d.iloc[30, d.columns.get_loc("high")] = d.iloc[30]["low"] - 1     # high < low
    d.iloc[40, d.columns.get_loc("volume")] = -5                      # negative volume
    d = pd.concat([d, d.iloc[[50]]])                                  # duplicate timestamp
    d = d.sort_index(ascending=False)                                 # unsorted
    clean, rep = validate(d)
    assert rep.unsorted
    assert rep.duplicates_removed == 1
    assert rep.nan_rows_removed == 1
    assert rep.non_positive_removed == 1
    assert rep.bad_ohlc_removed == 1
    assert rep.negative_volume_removed == 1
    assert len(clean) == 1001 - 1 - 4 and clean.index.is_monotonic_increasing
    assert clean.index.is_unique


def test_validate_does_not_mutate_input(winter):
    d = winter.iloc[:100].copy()
    d.iloc[5, 0] = np.nan
    before = d.copy()
    validate(d)
    pd.testing.assert_frame_equal(d, before)


def test_gaps_classified_and_not_filled(winter):
    clean, rep = validate(winter)
    assert len(clean) == len(winter)                                   # nothing invented
    kinds = rep.gaps["kind"].value_counts().to_dict()
    assert kinds.get("weekend", 0) >= 5                                # ~6 weekends
    d = winter.drop(winter.index[1000:1004])                           # 4 missing minutes
    g = find_gaps(d)
    assert ((g["kind"] == "short") & (g["missing_bars"] == 4)).any()


def test_spike_flagged_but_kept(winter):
    d = winter.iloc[:2000].copy()
    d.iloc[1500, d.columns.get_loc("close")] += 25                     # fat-finger style jump
    d.iloc[1500, d.columns.get_loc("high")] += 25
    clean, rep = validate(d)
    assert d.index[1500] in rep.spikes.index
    assert d.index[1500] in clean.index                                # kept


# ---------------------------------------------------------------- resample
def test_m5_values_match_manual(winter):
    m5 = resample(winter, "M5")
    t = m5.index[100]
    chunk = winter.loc[t: t + pd.Timedelta(minutes=4)]
    r = m5.loc[t]
    assert r["open"] == chunk["open"].iloc[0]
    assert r["close"] == chunk["close"].iloc[-1]
    assert r["high"] == chunk["high"].max() and r["low"] == chunk["low"].min()
    assert r["volume"] == chunk["volume"].sum()
    assert t.minute % 5 == 0                                           # open-time stamps


def test_volume_conserved_and_no_bars_in_closures(winter):
    for tf in ["M5", "M15", "M30", "H1", "H4"]:
        out = resample(winter, tf, drop_incomplete_last=False)
        assert out["volume"].sum() == winter["volume"].sum(), tf
        assert out.index.is_unique
    h1 = resample(winter, "H1")
    ny = h1.index.tz_convert(NY)
    assert not ((ny.day_of_week == 5)).any()                               # no Saturday bars
    assert not ((ny.hour == 17) & (ny.day_of_week < 5)).any()              # none in daily break


def test_last_incomplete_bar_dropped(winter):
    cut = winter.loc[: "2024-01-03 10:07:00"]                          # ends mid-M15 bar
    assert resample(cut, "M15").index[-1] == pd.Timestamp("2024-01-03 09:45", tz="UTC")
    assert resample(cut, "M15", drop_incomplete_last=False).index[-1] == pd.Timestamp("2024-01-03 10:00", tz="UTC")
    full = winter.loc[: "2024-01-03 10:14:00"]                         # M15 bar 10:00 is complete
    assert resample(full, "M15").index[-1] == pd.Timestamp("2024-01-03 10:00", tz="UTC")


def test_resample_rejects_unknown_timeframe(winter):
    with pytest.raises(ValueError):
        resample(winter, "M7")


def test_d1_rolls_at_1700_new_york():
    """D1 bars are stamped 17:00 NY (winter and summer) and 16:59 / 18:00 NY fall in adjacent days."""
    for start, end in [("2024-01-08", "2024-01-20"), ("2024-07-08", "2024-07-20")]:
        m1 = make_m1(start, end)
        d1 = resample(m1, "D1", drop_incomplete_last=False)
        ny = d1.index.tz_convert(NY)
        assert (ny.hour == 17).all() and (ny.minute == 0).all()
        assert d1["volume"].sum() == m1["volume"].sum()                # every M1 bar in exactly one D1 bar
        # Tuesday 16:59 NY belongs to Monday's session, Tuesday 18:00 NY to Tuesday's
        tue = pd.Timestamp(start, tz=NY) + pd.Timedelta(days=1)         # Tuesday
        before = (tue + pd.Timedelta(hours=16, minutes=59)).tz_convert("UTC")
        after = (tue + pd.Timedelta(hours=18)).tz_convert("UTC")
        d_before = d1.index[d1.index <= before][-1]
        d_after = d1.index[d1.index <= after][-1]
        assert d_after - d_before == pd.Timedelta(days=1)