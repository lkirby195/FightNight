"""names.py: aliases first, generational suffixes stripped on both sides,
hyphen handling and the fuzzy fallback unchanged."""
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
            ("jos", "Jose Aldo"), ("joe", "Joe Aldo")]


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


def test_finder_fuzzy_fallback_is_unchanged(monkeypatch):
    monkeypatch.setattr(names, "_aliases", {})
    find = names.make_finder(FIGHTERS)
    assert find("Patricio Freire") is None       # no alias: surname Freire, but the
                                                 # first-name guard blocks the brother
    assert find("Jos Aldo") == "jos"             # surname + first three letters, unique
    assert find("Jo Aldo") is None               # "jo" is neither "jos" nor "joe"
    assert find("Michael Aswell") == "asw"       # the suffix strip alone makes it exact
    assert find("") is None
