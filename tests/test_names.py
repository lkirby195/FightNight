"""names.py: aliases first, generational suffixes stripped on both sides,
hyphen handling, no fuzzy auto-accept, and the fail-loud resolver (ruled
2026-10-22): a spelling near-miss raises, a debut resolves to None."""
import pytest

import names


@pytest.fixture(autouse=True)
def seed_aliases(monkeypatch):
    monkeypatch.setattr(names, "_aliases", {"Patricio Freire": "Patricio Pitbull",
                                             "Michael Aswell": "Michael Aswell Jr."})


def test_norm_strips_generational_suffixes():
    assert names.norm("Michael Aswell Jr.") == "michael aswell"
    assert names.norm("Sean King III") == "sean king"
    assert names.norm("Kai Kamaka III") == names.norm("Kai Kamaka")
    assert names.norm("Lance Gibson Jr.") == "lance gibson"
    assert names.norm("Roger Sr.") == "roger"           # single-token name kept
    assert names.norm("Jr.") == "jr"                    # nothing to strip against
    assert names.lastn("Michael Aswell Jr.") == "aswell"
    assert names.lastn("Sean King III") == "king"


def test_hyphen_handling_is_unchanged():
    assert names.norm("Waldo Cortes-Acosta") == "waldo cortes acosta"
    assert names.norm("Sangcha-An", "") == names.norm("Sangcha'an", "") == "sangchaan"
    assert names.norm("Al-Hassan", "") == "alhassan"
    assert names.norm("Joo Sang Yoo").replace(" ", "") == names.norm("JooSang Yoo").replace(" ", "")
    assert names.lastn("Waldo Cortes-Acosta") == "acosta"
    assert names.lastn("Waldo Cortes-Acosta", "") == "cortesacosta"


def test_aliases_apply_to_bfo_names_only():
    assert names.bfo_norm("Patricio Freire") == "patricio pitbull"
    assert names.bfo_lastn("Patricio Freire") == "pitbull"
    assert names.bfo_norm("Michael Aswell") == "michael aswell"     # suffix goes too
    assert names.norm("Patricio Freire") == "patricio freire"       # UFC Stats side untouched
    assert names.bfo_norm("Willamy Freire") == "willamy freire"     # the brother is not aliased
    assert names.alias("  Patricio Freire ") == "Patricio Pitbull"


def test_load_aliases_reads_the_seed_file():
    a = names.load_aliases()
    assert a["Patricio Freire"] == "Patricio Pitbull"
    assert a["Michael Aswell"] == "Michael Aswell Jr."
    assert names.load_aliases("does/not/exist.csv") == {}


FIGHTERS = [("pit", "Patricio Pitbull"), ("wil", "Willamy Freire"),
            ("asw", "Michael Aswell Jr."), ("yoo", "JooSang Yoo"),
            ("cho", "Dooho Choi"), ("kin", "Sean King III"),
            ("wca", "Waldo Cortes Acosta"), ("san", "Sangcha'an"),
            ("jos", "Jose Aldo"), ("joe", "Joe Aldo"),
            ("luc", "Iasmin Lucindo"), ("cal", "Nicolle Caliari")]


def test_finder_resolves_aliases_and_suffixes():
    find = names.make_finder(FIGHTERS)
    assert find("Patricio Freire") == "pit"      # alias
    assert find("Willamy Freire") == "wil"       # exact: the brother stays himself
    assert find("Michael Aswell") == "asw"       # alias + suffix
    assert find("Sean King") == "kin"            # suffix on the UFC Stats side only
    assert find("Joo Sang Yoo") == "yoo"         # joined form
    assert find("Doo Ho Choi") == "cho"
    assert find("Waldo Cortes-Acosta") == "wca"  # hyphen
    assert find("Sangcha-An") == "san"


def test_finder_has_no_fuzzy_fallback(monkeypatch):
    monkeypatch.setattr(names, "_aliases", {})
    find = names.make_finder(FIGHTERS)
    assert find("Patricio Freire") is None       # no alias: no guess at the brother
    assert find("Jos Aldo") is None              # surname + three letters used to pass
    assert find("Jo Aldo") is None
    assert find("Michael Aswell") == "asw"       # the suffix strip alone makes it exact
    assert find("Lasmin Lucindo") is None        # one letter off: not a match
    assert find("") is None


def test_near_misses_are_ranked_and_thresholded():
    c = names.near_misses("Lasmin Lucindo", FIGHTERS)
    assert c and c[0][:2] == ("luc", "Iasmin Lucindo") and c[0][2] >= names.NEAR_MISS
    assert names.near_misses("Nicolle Caliari", FIGHTERS)[0][0] == "cal"   # an exact name is 1.0
    assert names.near_misses("Alex Perez", FIGHTERS) == []
    assert names.near_misses("Su Mudaerji", [("su", "Sumudaerji")])[0][0] == "su"  # joined form


def test_resolver_raises_on_the_lucindo_case_before_the_alias(monkeypatch):
    """UFC 333: BFO 'Lasmin Lucindo' vs 'Nicolle Caliari'. Caliari resolves,
    Lucindo does not, and Iasmin Lucindo is on file: a spelling miss, not a
    debut. Nothing is guessed; the error names the candidate."""
    monkeypatch.setattr(names, "_aliases", {})
    resolve = names.make_resolver(FIGHTERS)
    with pytest.raises(names.NameJoinError) as e:
        resolve("Lasmin Lucindo", "Nicolle Caliari")
    msg = str(e.value)
    assert "Lasmin Lucindo" in msg and "Iasmin Lucindo" in msg and "name_aliases.csv" in msg
    with pytest.raises(names.NameJoinError):
        resolve("Nicolle Caliari", "Lasmin Lucindo")        # either side


def test_resolver_joins_the_lucindo_case_after_the_alias(monkeypatch):
    monkeypatch.setattr(names, "_aliases", {"Lasmin Lucindo": "Iasmin Lucindo"})
    resolve = names.make_resolver(FIGHTERS)
    assert resolve("Lasmin Lucindo", "Nicolle Caliari") == ("luc", "cal")
    assert resolve("Nicolle Caliari", "Lasmin Lucindo") == ("cal", "luc")


def test_resolver_treats_an_unknown_name_without_a_near_miss_as_a_debut(monkeypatch):
    monkeypatch.setattr(names, "_aliases", {})
    resolve = names.make_resolver(FIGHTERS)
    assert resolve("Asaf Chopurov", "Nicolle Caliari") == (None, "cal")   # one debutant
    assert resolve("Asaf Chopurov", "Bruno Nobody") == (None, None)      # neither on file
    # both unknown never raises, even when one is a near-miss: a likely debut bout
    assert resolve("Lasmin Lucindo", "Bruno Nobody") == (None, None)


def test_a_distinct_person_row_silences_the_near_miss(monkeypatch):
    # data/name_aliases.csv: "Lasmin Lucindo," (empty ufcstats_name) = a different person
    monkeypatch.setattr(names, "_aliases", {"Lasmin Lucindo": ""})
    assert names.distinct("Lasmin Lucindo") and not names.distinct("Nicolle Caliari")
    assert names.alias("Lasmin Lucindo") == "Lasmin Lucindo"
    resolve = names.make_resolver(FIGHTERS)
    assert resolve("Lasmin Lucindo", "Nicolle Caliari") == (None, "cal")


def test_near_miss_pair_for_the_bout_level_join(monkeypatch):
    monkeypatch.setattr(names, "_aliases", {})
    hit = names.near_miss_pair("Lasmin Lucindo", "Nicolle Caliari",
                               "Iasmin Lucindo", "Nicolle Caliari")
    assert hit[:2] == ("Lasmin Lucindo", "Iasmin Lucindo") and hit[2] >= names.NEAR_MISS
    assert names.near_miss_pair("Nicolle Caliari", "Lasmin Lucindo",
                                "Nicolle Caliari", "Iasmin Lucindo")[1] == "Iasmin Lucindo"
    assert names.near_miss_pair("Lasmin Lucindo", "Nicolle Caliari",
                                "Iasmin Lucindo", "Nicolle Caliari")[0] == "Lasmin Lucindo"
    assert names.near_miss_pair("Asaf Chopurov", "Colby Thicknesse",
                                "Colby Thicknesse", "Someone Else") is None
    assert names.near_miss_pair("A Person", "B Person", "C Person", "D Person") is None
