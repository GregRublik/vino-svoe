import hashlib

from utils.ids import stable_int_id


def test_stable_int_id_is_deterministic():
    assert stable_int_id("Вино.webp") == stable_int_id("Вино.webp")


def test_stable_int_id_matches_sha256_prefix():
    filename = "Вино_красное_сухое_Мерло.webp"
    expected = int(hashlib.sha256(filename.encode("utf-8")).hexdigest()[:16], 16)
    assert stable_int_id(filename) == expected


def test_stable_int_id_differs_between_filenames():
    assert stable_int_id("a.webp") != stable_int_id("b.webp")


def test_stable_int_id_is_positive_int():
    assert isinstance(stable_int_id("x.webp"), int)
    assert stable_int_id("x.webp") > 0
