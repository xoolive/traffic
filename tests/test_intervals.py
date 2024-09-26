import pandas as pd
from traffic.core.intervals import Interval, IntervalCollection

dates = pd.date_range("2022-03-21 11:10", "2022-03-21 11:20", freq="1T")
h0, h1, h2, h3, h4, h5, h6, h7, h8, *_ = dates

i1 = Interval(h0, h2)
i2 = Interval(h3, h6)
i3 = Interval(h6, h8)
i4 = Interval(h1, h5)
i5 = Interval(h0, h8)

c1 = IntervalCollection(start=[h0, h1], stop=[h2, h5])
c2 = IntervalCollection(start=[h1, h0], stop=[h5, h2])
c3 = IntervalCollection(start=[h1, h0], stop=[h5, h8])
c4 = IntervalCollection(start=[h0, h1], stop=[h8, h5])
c5 = IntervalCollection(start=[h0, h3], stop=[h2, h6])
c6 = IntervalCollection(start=[h3, h0], stop=[h6, h2])
c7 = IntervalCollection(start=[h7], stop=[h8])


class TestInterval:
    def test_eq(self) -> None:
        assert i1 is not None
        assert i1 == i1
        assert i1 != i2
        assert i1 != []

    def test_duration(self) -> None:
        assert i1.duration() == pd.Timedelta("2 minutes")


class TestIntervalAdd:
    def test_partialsup(self) -> None:
        assert i1 + i4 == IntervalCollection(start=[h0, h1], stop=[h2, h5])

    def test_totalsup(self) -> None:
        assert i2 + i5 == IntervalCollection(start=[h0, h3], stop=[h8, h6])

    def test_nosup(self) -> None:
        assert i1 + i2 == IntervalCollection(start=[h0, h3], stop=[h2, h6])


class TestIntervalSubtract:
    def test_partialsupleft(self) -> None:
        assert i1 - i4 == IntervalCollection(h0, h1)

    def test_partialsupright(self) -> None:
        assert i2 - i4 == IntervalCollection(h5, h6)

    def test_totalsup(self) -> None:
        assert i2 - i5 is None

    def test_partialsupmiddle(self) -> None:
        assert i5 - i2 == IntervalCollection(start=[h0, h6], stop=[h3, h8])

    def test_nosup_left(self) -> None:
        assert i1 - i2 == IntervalCollection(i1)

    def test_nosup_right(self) -> None:
        assert i2 - i1 == IntervalCollection(i2)

    def test_equal(self) -> None:
        assert i1 - i1 is None


class TestIntervalUnion:
    def test_partialsupleft(self) -> None:
        assert i1 | i4 == IntervalCollection(h0, h5)

    def test_partialsupright(self) -> None:
        assert i2 | i4 == IntervalCollection(h1, h6)

    def test_totalsup(self) -> None:
        assert i2 | i5 == IntervalCollection(h0, h8)

    def test_partialsupmiddle(self) -> None:
        assert i5 | i2 == IntervalCollection(h0, h8)

    def test_nosup_left(self) -> None:
        assert i1 | i2 == IntervalCollection(start=[h0, h3], stop=[h2, h6])

    def test_nosup_right(self) -> None:
        assert i2 | i1 == IntervalCollection(start=[h0, h3], stop=[h2, h6])

    def test_equal(self) -> None:
        assert i1 | i1 == IntervalCollection(h0, h2)


class TestIntervalIntersection:
    def test_partialsupleft(self) -> None:
        assert i1 & i4 == Interval(h1, h2)

    def test_partialsupright(self) -> None:
        assert i2 & i4 == Interval(h3, h5)

    def test_totalsup(self) -> None:
        assert i2 & i5 == Interval(h3, h6)

    def test_partialsupmiddle(self) -> None:
        assert i5 & i2 == Interval(h3, h6)

    def test_nosup_left(self) -> None:
        assert i1 & i2 is None

    def test_nosup_right(self) -> None:
        assert i2 & i1 is None

    def test_equal(self) -> None:
        assert i1 & i1 == Interval(h0, h2)


class TestCollection:
    def test_duration_simple(self) -> None:
        assert c7.total_duration() == pd.Timedelta(seconds=60)

    def test_consolidate(self) -> None:
        assert c1.consolidate() == IntervalCollection(h0, h5)
        assert c2.consolidate() == IntervalCollection(h0, h5)
        assert c3.consolidate() == IntervalCollection(h0, h8)
        assert c5.consolidate() == c5
        assert c6.consolidate() == c6

        other = IntervalCollection(i1, i2, i3, i4)
        res = other.consolidate()
        assert res is not None
        assert next(iter(res)) == Interval(h0, h8)


class TestCollectionAdd:
    def test_same(self) -> None:
        assert c1 + c1 == (
            IntervalCollection(start=[h0, h0, h1, h1], stop=[h2, h2, h5, h5])
        )

    def test_diffsup(self) -> None:
        assert c1 + c4 == (
            IntervalCollection(start=[h0, h0, h1, h1], stop=[h2, h8, h5, h5])
        )

    def test_diffnosup(self) -> None:
        assert c1 + c7 == IntervalCollection([h0, h1, h7], [h2, h5, h8])


class TestCollectionSub:
    def test_same(self) -> None:
        assert c1 == c1
        assert c1 - c1 is None

    def test_partminustot(self) -> None:
        assert c1 - c4 is None

    def test_smthminusinv(self) -> None:
        assert c5 - c6 is None

    def test_supleft(self) -> None:
        assert c3 - c2 == IntervalCollection(h5, h8)

    def test_supright(self) -> None:
        assert c3 - c7 == IntervalCollection(h0, h7)

    def test_hole(self) -> None:
        assert c6 - c2 == IntervalCollection(h5, h6)
        assert c6 - IntervalCollection(
            start=[h0, h4], stop=[h1, h5]
        ) == IntervalCollection(
            Interval(h1, h2), Interval(h3, h4), Interval(h5, h6)
        )

    def test_nosup(self) -> None:
        # no overlap between ic01 and ic07, nothing to substract
        assert c1 - c7 == c1.consolidate()


class TestCollectionUnion:
    def test_sup(self) -> None:
        assert c1 | c2 == IntervalCollection(h0, h5)

    def test_nosup(self) -> None:
        assert c5 | c7 == (c5 + c7)


class TestCollectionIntersection:
    def test_sup(self) -> None:
        assert c1 & c2 == IntervalCollection(h0, h5)

    def test_nosup(self) -> None:
        assert c5 & c7 is None
