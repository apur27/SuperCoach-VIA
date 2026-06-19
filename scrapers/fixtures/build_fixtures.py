"""
Regenerate the persisted free-agency fixtures from hand-verified ground truth.

Both source sites (AFL.com.au, ZeroHanger) are Cloudflare-fronted and 403 a plain
``requests`` call, so the scraper falls back to these fixtures. The verified content
below was confirmed on 2026-06-19:

  AFL.com.au 2026 free agents list (status authoritative):
    https://www.afl.com.au/news/1484077/2026afl-free-agentslist
  ZeroHanger off-contract 2026 (broader pool -- only the notable additional names
  NOT already on the FA list are encoded here):
    https://www.zerohanger.com/afl/players/off-contract-2026/

Run: python scrapers/fixtures/build_fixtures.py
This emits afl_fa_2026.html and zerohanger_offcontract_2026.html alongside it,
in the exact structure free_agency_scraper.parse_* expects.
"""

import os

# club -> [(player_name, "Restricted" | "Unrestricted")]
AFL_FA = {
    "Adelaide": [("James Borlase", "Restricted"), ("Jordon Butts", "Unrestricted"),
                 ("Mitchell Hinge", "Restricted"), ("Chayce Jones", "Unrestricted"),
                 ("Lachlan Sholl", "Unrestricted"), ("Taylor Walker", "Unrestricted"),
                 ("Tyler Welsh", "Restricted")],
    "Brisbane": [("Zac Bailey", "Restricted"), ("Darragh Joyce", "Unrestricted"),
                 ("Ryan Lester", "Unrestricted"), ("Lincoln McCarthy", "Unrestricted"),
                 ("Lachie Neale", "Unrestricted"), ("Dayne Zorko", "Unrestricted")],
    "Carlton": [("Wade Derksen", "Restricted"), ("Francis Evans", "Restricted"),
                ("Nick Haynes", "Unrestricted"), ("Elijah Hollands", "Restricted"),
                ("Mitchell McGovern", "Unrestricted"), ("Nicholas Newman", "Unrestricted"),
                ("Zachary Williams", "Unrestricted")],
    "Collingwood": [("Jack Crisp", "Unrestricted"), ("Jeremy Howe", "Unrestricted"),
                    ("Tim Membrey", "Unrestricted"), ("Scott Pendlebury", "Unrestricted"),
                    ("Steele Sidebottom", "Unrestricted")],
    "Essendon": [("Jade Gresham", "Restricted"), ("Matt Guelfi", "Unrestricted"),
                 ("Liam McMahon", "Restricted"), ("Jaxon Prior", "Restricted"),
                 ("Will Setterfield", "Restricted")],
    "Fremantle": [("Bailey Banfield", "Restricted"), ("Oscar McDonald", "Restricted"),
                  ("Samuel Sturt", "Unrestricted"), ("Sam Switkowski", "Unrestricted"),
                  ("Corey Wagner", "Restricted")],
    "Geelong": [("Jed Bews", "Restricted"), ("Mark Blicavs", "Unrestricted"),
                ("Patrick Dangerfield", "Unrestricted"), ("Jake Kolodjashnij", "Unrestricted"),
                ("Jack Martin", "Restricted"), ("Keighton Matofai-Forbes", "Restricted"),
                ("Mark O'Connor", "Unrestricted"), ("Rhys Stanley", "Unrestricted"),
                ("George Stevens", "Restricted")],
    "Gold Coast": [("Ben King", "Restricted"), ("Oscar Adams", "Restricted"),
                   ("Caleb Graham", "Unrestricted"), ("Nick Holman", "Restricted"),
                   ("Ben Jepsen", "Restricted"), ("Lachlan Weller", "Unrestricted"),
                   ("Jarrad Witts", "Unrestricted")],
    "GWS": [("Kieren Briggs", "Restricted"), ("Stephen Coniglio", "Unrestricted"),
            ("Toby Greene", "Unrestricted"), ("Jayden Laverde", "Restricted"),
            ("Jake Stringer", "Restricted"), ("Conor Stone", "Restricted")],
    "Hawthorn": [("Jarman Impey", "Restricted"), ("Jack Gunston", "Unrestricted"),
                 ("Mitchell Lewis", "Unrestricted"), ("Harry Morrison", "Unrestricted"),
                 ("Flynn Perez", "Restricted")],
    "Melbourne": [("Bayley Fritsch", "Restricted"), ("Tom Campbell", "Restricted"),
                  ("Kade Chandler", "Unrestricted"), ("Jai Culley", "Restricted"),
                  ("Tom McDonald", "Unrestricted"), ("Jake Melksham", "Restricted"),
                  ("Christian Salem", "Unrestricted"), ("Thomas Sparrow", "Unrestricted")],
    "North Melbourne": [("Aidan Corr", "Restricted"), ("Luke McDonald", "Unrestricted"),
                        ("Toby Pink", "Restricted"), ("Bailey Scott", "Unrestricted")],
    "Port Adelaide": [("Zak Butters", "Restricted"), ("Darcy Byrne-Jones", "Unrestricted"),
                      ("Oliver Wines", "Unrestricted")],
    "Richmond": [("Nathan Broad", "Restricted"), ("Thomas Lynch", "Unrestricted"),
                 ("Dion Prestia", "Unrestricted")],
    "St Kilda": [("Ryan Byrnes", "Restricted"), ("Jack Carroll", "Restricted"),
                 ("Patrick Said", "Restricted"), ("Liam Stocker", "Restricted")],
    "Sydney": [("Joel Amartey", "Restricted"), ("Harry Cunningham", "Unrestricted"),
               ("Joel Hamling", "Restricted"), ("Jake Lloyd", "Restricted"),
               ("Justin McInerney", "Unrestricted"), ("Dane Rampe", "Unrestricted")],
    "West Coast": [("Bailey Williams", "Restricted"), ("Thomas Cole", "Unrestricted"),
                   ("Jamie Cripps", "Unrestricted"), ("Matthew Flynn", "Unrestricted"),
                   ("Harry Schoenberg", "Restricted")],
    "Western Bulldogs": [("Oskar Baker", "Restricted"), ("Lachlan Bramble", "Restricted"),
                         ("Ryan Gardner", "Restricted"), ("Tom Liberatore", "Unrestricted"),
                         ("Buku Khamis", "Unrestricted"), ("Lachlan McNeil", "Restricted")],
}

# ZeroHanger broader off-contract pool -- ONLY notable names NOT already on the FA
# list above. "Tom J. Lynch" (Richmond) is intentionally omitted: it is the same
# player as "Thomas Lynch" already on Richmond's FA list, and including it would
# create a false duplicate. Ryan Gardner (WB) is on the FA list already; dedup
# drops the ZeroHanger copy.
ZEROHANGER_EXTRA = {
    "Port Adelaide": ["Aliir Aliir", "Ivan Soldo", "Jordon Sweet", "Brandon Zerk-Thatcher",
                      "Jackson Mead", "Ollie Lord", "Lachlan Jones"],
    "Gold Coast": ["Jamarra Ugle-Hagan", "Jed Walter", "John Noble"],
    "Carlton": ["Adam Saad", "Lachie Fogarty", "Jordan Boyd"],
    "North Melbourne": ["Callum Coleman-Jones", "Zane Duursma", "Cooper Harvey", "Luke Parker"],
    "Hawthorn": ["Calsher Dear", "Karl Amon"],
    "GWS": ["Jesse Hogan"],
    "Western Bulldogs": ["Adam Treloar", "Ryan Gardner"],
    "Richmond": ["Toby Nankervis", "Tyler Sonsie"],
}

_HERE = os.path.dirname(os.path.abspath(__file__))


def build_afl_html():
    sections = []
    for club, players in AFL_FA.items():
        rows = "".join(
            f'<tr><td class="player">{n}</td><td class="status">{s}</td></tr>'
            for n, s in players
        )
        sections.append(
            f'<div class="club-section" data-club="{club}">'
            f'<h3 class="club-name">{club}</h3><table>{rows}</table></div>'
        )
    return ('<!doctype html><html><head><title>2026 AFL free agents</title></head>'
            '<body><div class="free-agents-list">' + "".join(sections)
            + "</div></body></html>\n")


def build_zh_html():
    blocks = []
    for club, names in ZEROHANGER_EXTRA.items():
        lis = "".join(f"<li>{n}</li>" for n in names)
        blocks.append(
            f'<div class="club" data-club="{club}"><h3>{club}</h3><ul>{lis}</ul></div>')
    return ('<!doctype html><html><head><title>Off-contract 2026</title></head>'
            '<body><div class="off-contract">' + "".join(blocks)
            + "</div></body></html>\n")


def main():
    with open(os.path.join(_HERE, "afl_fa_2026.html"), "w", encoding="utf-8") as fh:
        fh.write(build_afl_html())
    with open(os.path.join(_HERE, "zerohanger_offcontract_2026.html"), "w",
              encoding="utf-8") as fh:
        fh.write(build_zh_html())
    n_afl = sum(len(v) for v in AFL_FA.values())
    n_zh = sum(len(v) for v in ZEROHANGER_EXTRA.values())
    print(f"wrote AFL fixture ({n_afl} FA players) and ZeroHanger fixture ({n_zh} extra)")


if __name__ == "__main__":
    main()
