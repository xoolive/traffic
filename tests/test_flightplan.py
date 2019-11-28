# fmt: off
from typing import List, cast

from traffic.core.flightplan import (Airway, CoordinatePoint, Direct,
                                     FlightPlan, Point, SpeedLevel,
                                     _ElementaryBlock)

# fmt: on

flightplans: List[FlightPlan] = [
    # Local flight plan, LFBO to EGLL: easy
    FlightPlan(
        "N0456F340 LACOU5A LACOU UM184 CNA UN863 MANAK UY110 REVTU UP87 ROXOG ROXOG1H",  # noqa: E501
        "LFBO",
        "EGLL",
    ),
    # With speed and unit changes, LFPG to RJBB
    FlightPlan(
        "N0495F320 RANUX3D RANUX UN858 VALEK/N0491F330 UM163 DIK UN853 ARCKY DCT NVO DCT BERIM DCT BIKRU/N0482F350 DCT VEDEN DCT NETNA DCT TUKMA L738 MEGAS/K0893F350 P989 TRAFA R492 ESAPA R959 LERTA G814 ABODI/K0883F370 G814 SADER T580 ADMUR G117 XV R229 BA R104 ABK A308 TR P982 DARNO/K0898S1130 A575 UDA B339 POLHO/K0905S1130 W33 UKDUM W201 UNSEK A326 DONVO G597 AGAVO/N0489F370 Y644 EGOBA G597 LANAT/N0491F370 Y597 MIHOU Y361 SAEKI Y36 ALISA ALISAB",  # noqa: E501
        "LFPG",
        "RJBB",
    ),
    # Europe (EHAM) to US, with coordinate points
    FlightPlan(
        "0458F320 BERGI UL602 SUPUR UP1 GODOS P1 ROLUM P13 ASKAM L7 SUM DCT PEMOS/M079F320 DCT 62N010W 63N020W 63N030W 64N040W 64N050W DCT CLAVY DCT BERUS DCT FEDDY DCT 61N070W 60N080W 57N090W/N0456F340 55N096W 53N100W DCT GGW/N0457F360 DCT BIL DCT JAC NORDK6",  # noqa: E501
        "EHAM",
        "KSLC",
    ),
    # With some \n in between
    FlightPlan(
        "N0427F230 DET1J DET L6 DVR L9 KONAN/N0470F350 UL607 MATUG UZ660 BOREP DCT ENITA DCT ETVIS DCT ABUDO/N0470F370 Z37 BUDEX DCT PESAT DCT TEGRI L605 NEKUL P975 ARTAT UP975 ERGUN UL124 ASVOD/N0468F390 UL124 KONUK UM860 RENGI UT253 OTKEP UM688 RATVO/N0470F390 UM688 SIDAD/N0475F390 UP975 SESRA M677 RABAP UM677 OBNET M677 LOVEM L562 SERSA P307 PARAR N571 SUGID",  # noqa: E501
        "EGLL",
        "VABB",
    ),
    # LFPG, confusion between AS and *AS
    FlightPlan(
        "N0463F350 ERIXU3B ERIXU UN860 ETAMO UZ271 ADEKA UT18 AMLIR/N0461F370 UT18 BADAM UZ151 FJR UM731 DIVKO UM989 BALEN UM998 KAMER/N0465F370 UM998 BOD UR978 ATAFA/N0469F370 UR978 ERKEL/N0473F370 UR978 AS UA604 KOKAM KOKAM1V",  # noqa: E501
        "LFPG",
        "FKYS",
    ),
    # EDDF to MMUN
    FlightPlan(
        "N0459F320 OBOKA UZ29 TORNU DCT RAVLO Y70 OTBED L60 PENIL M144 BAGSO DCT RINUS DCT GISTI/M079F330 DCT MALOT/M079F340 DCT 54N020W 55N030W 54N040W 51N050W DCT ALLRY/N0463F360 DCT YQX DCT ELERI/M079F360 DCT AVAST DCT DRYED M201 TILED DCT ONGOT/M079F380 M202 OMALA DCT SNAGY/N0453F380 DCT RAMJT A699 PERMT DCT FIS B646 CANOA UB879 NOSAT NOSAT1A",  # noqa: E501
        "EDDF",
        "MMUN",
    ),
]


def test_direct():
    assert Direct.valid("DCT")
    assert not Direct.valid("N0450F340")
    assert type(_ElementaryBlock.parse("DCT")) is Direct


def test_speedlevel():
    assert SpeedLevel.valid("N0450F340")
    sl = cast(SpeedLevel, _ElementaryBlock.parse("N0450F340"))
    assert sl is not None
    assert sl.speed == 450
    assert sl.altitude == 340


def test_airway():
    assert Airway.valid("L888")
    assert not Airway.valid("123456789")
    a = cast(Airway, _ElementaryBlock.parse("L888"))
    r = a.get()
    assert r is not None
    assert "SANLI" in r.navaids
    assert r.project_shape().length > 2.9e6


def test_point():
    assert Point.valid("NARAK")
    p = Point("NARAK")
    assert p.name == "NARAK"
    n = p.get()
    assert n is not None
    assert n.name == "NARAK"

    assert CoordinatePoint.valid("61N070W")
    c = CoordinatePoint("61N070W")
    assert c.lat == 61
    assert c.lon == -70
    n = c.get()
    assert n is not None
    assert n.latlon == (61, -70)


def test_flightplan():
    for fp in flightplans:
        elts = fp.decompose()

        assert any(isinstance(p, SpeedLevel) for p in elts)
        # we can parse everything
        assert all(p is not None for p in elts)
        for cur_, next_ in zip(elts, elts[1:]):
            # never two consecutive airways
            assert not isinstance(cur_, Airway) or not isinstance(next_, Airway)
            # never two consecutive navaids (coordinate points are ok though)
            assert not isinstance(cur_, Point) or not isinstance(next_, Point)
