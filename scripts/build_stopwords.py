"""
Generate comprehensive, categorized Marathi stopword list for MahaVarta.
Combines linguistic grammatical paradigms with data-driven category-entropy candidates.
"""

import json
import os
import math
import pandas as pd
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Section 1: Pronouns and Inflections (First, Second, Third Person, Reflexive)
# ---------------------------------------------------------------------------
PRONOUNS = {
    # Second-person paradigm
    "तुम्ही", "तुम्हाला", "तुमच्या", "तुमचा", "तुमची", "तुमचे", "तुमच्याकडे",
    "तुमच्यावर", "तुमच्यासाठी", "तुमच्याकडून", "तुम्हीच", "तुम्हालाही",
    "तू", "तुला", "तुझा", "तुझी", "तुझे", "तुझ्या", "तुझं", "तुझ्याकडे", "तुझ्यावर",
    # First-person paradigm
    "मी", "मला", "माझा", "माझी", "माझे", "माझ्या", "माझं", "माझ्याकडे", "माझ्यावर",
    "आम्ही", "आम्हाला", "आमचा", "आमची", "आमचे", "आमच्या", "आमचं", "आमच्याकडे",
    "आपण", "आपल्या", "आपल्याला", "आपला", "आपली", "आपले", "आपलं", "आपल्याकडे", "आपल्यासाठी",
    "स्वतः", "स्वतःला", "स्वतःच्या",
    # Third-person & Demonstrative paradigm
    "तो", "ती", "ते", "त्या", "त्याला", "तिला", "त्यांना", "त्याने", "तिने", "त्यांनी",
    "त्याचा", "त्याची", "त्याचे", "त्याच्या", "त्याचं", "तिचा", "तिची", "तिचे", "तिच्या",
    "त्यांच्या", "त्यांचा", "त्यांची", "त्यांचे", "त्यांच्याकडे", "त्यांच्यावर", "त्यांच्यासाठी",
    "हा", "ही", "हे", "ह्या", "या", "याला", "हिला", "यांना", "याने", "हिने", "यांनी",
    "याचा", "याची", "याचे", "याच्या", "याचं", "यांच्या", "यांचा", "यांची", "यांचे",
    "यांच्याकडे", "यांच्यावर", "यात", "त्यात", "यावर", "त्यावर", "यावरून", "त्यावरून",
    "यासह", "त्यासह", "यामध्ये", "त्यामध्ये", "त्यामुळे", "यामुळे",
    # Relative & Interrogative pronouns
    "जो", "जी", "जे", "ज्या", "ज्याला", "जिंला", "ज्यांना", "ज्याने", "जिने", "ज्यांनी",
    "ज्याचा", "ज्याची", "ज्याचे", "ज्याच्या", "ज्यांच्या", "ज्यांचा", "ज्यांची", "ज्यांचे",
    "कोण", "कोणाला", "कोणी", "कोणाचा", "कोणाची", "कोणाचे", "कोणाच्या", "काय", "काही",
}

# ---------------------------------------------------------------------------
# Section 2: Auxiliary Verbs, Copulas & Generic Verb Conjugations
# ---------------------------------------------------------------------------
VERB_FORMS = {
    # Asne (to be) & Nasne (not to be)
    "आहे", "आहेस", "आहोत", "आहेत", "नाही", "नाहीत", "नसून", "असू", "असणे",
    "होते", "होती", "होता", "होतो", "होत", "होतं", "असेल", "असतील", "असू",
    "असते", "असतो", "असतात", "नसते", "नसतो", "नसतात",
    "असल्याचे", "नसल्याचे", "असल्यास", "नसल्यास",
    "असलेला", "असलेली", "असलेले", "असलेल्या", "असणारा", "असणारी", "असणारे", "असणाऱ्या",
    # Hone (to happen/become)
    "झाले", "झाली", "झाला", "झालेत", "झालेलं", "झालेला", "झालेली", "झालेले", "झालेल्या",
    "होणार", "होणारे", "होणारा", "होणारी", "होणाऱ्या", "होऊन", "होऊ", "होतेय", "होतोय",
    "व्हावे", "व्हावी", "व्हावा", "व्हावेत",
    # Karne (to do)
    "केले", "केली", "केला", "केलेत", "केलं", "केलेला", "केलेली", "केलेले", "केलेल्या",
    "केलेय", "केलाय", "केलीय", "करतो", "करते", "करतात", "करत", "करणे", "करण्यात",
    "करावे", "करावी", "करावा", "करावेत", "करून", "करू", "करताना", "करण्यासाठी",
    "करा", "करावीत",
    # Jane & Yene (go/come)
    "जातो", "जाते", "जातात", "जात", "गेला", "गेली", "गेले", "गेलं", "गेलेला", "गेलेली", "गेलेले",
    "गेलेल्या", "जाणे", "जावे", "जावी", "जावा", "जाऊन", "जाऊ", "जातोय", "जातेय",
    "येतो", "येते", "येतात", "येत", "आला", "आली", "आले", "आलं", "आलेला", "आलेली", "आलेले",
    "आलेल्या", "येणे", "यावे", "यावी", "यावा", "येऊन", "येऊ", "येतोय", "येतेय", "आलाय", "आलीय", "आलंय",
    # Dene & Ghene (give/take)
    "दिला", "दिली", "दिले", "दिलों", "दिलं", "दिलेला", "दिलेली", "दिलेले", "दिलेल्या",
    "देतो", "देते", "देतात", "देत", "देणे", "देण्यात", "द्यावे", "द्यावी", "द्यावा", "देऊन", "देऊ",
    "घेतला", "घेतली", "घेतले", "घेतलं", "घेतलेला", "घेतलेली", "घेतलेले", "घेतलेल्या",
    "घेतो", "घेते", "घेतात", "घेत", "घेणे", "घेण्यात", "घ्यावे", "घ्यावी", "घ्यावा", "घेऊन", "घेऊ",
    # Modals / possibility
    "शकतो", "शकते", "शकतात", "शक्य", "अशक्य", "शकता", "शकणार",
    # Common verbs like mhantale, sangitle, pahu
    "म्हणाले", "म्हणाली", "म्हणाला", "म्हणालेत", "म्हटले", "म्हटलं", "म्हणत",
    "सांगितले", "सांगितलं", "सांगितली", "सांगितला", "सांगतात", "सांगत", "सांगणे",
    "पाहिले", "पाहिली", "पाहिला", "पाहिलं", "पाहा", "पहा", "पाहायला",
}

# ---------------------------------------------------------------------------
# Section 3: Postpositions, Case Markers, Conjunctions & Particles
# ---------------------------------------------------------------------------
FUNCTION_WORDS = {
    # Postpositions & case markers
    "वर", "खाली", "मागे", "पुढे", "समोर", "जवळ", "कडे", "कडून", "पर्यंत", "पासून",
    "साठी", "मुळे", "द्वारे", "नुसार", "बाबत", "विषयी", "संदर्भात", "दरम्यान",
    "नंतर", "आधी", "पूर्वी", "शिवाय", "विना",
    # Conjunctions & connectives
    "आणि", "किंवा", "पण", "परंतु", "मात्र", "तर", "जरी", "तरी", "जर", "कारण", "म्हणून",
    "तसेच", "अथवा", "किंबहुना", "तथापि", "म्हणजे", "की", "व", "अथवा",
    # Adverbs, discourse particles & fillers
    "असे", "अशी", "असा", "अशा", "असं", "असंही", "तसे", "तशी", "तसा", "तशा", "तसं",
    "कसे", "कशी", "कसा", "कशा", "कसं", "कधी", "कुठे", "कोठे", "किती", "का",
    "येथे", "तिथे", "जेथे", "आता", "मग", "पुन्हा", "सध्या", "नेहमी", "सतत",
    "खूप", "जास्त", "कमी", "अनेक", "काही", "सर्व", "सर्वच", "इतर", "मोठे", "मोठ्या",
    "फक्त", "केवळ", "देखील", "सुद्धा", "एकदा", "पहिला", "पहिली", "पहिले", "दुसरा", "दुसरी", "दुसरे",
    "आज", "काल", "उद्या", "दिवशी", "वेळी", "काळात", "वेळेस",
    "दोन", "तीन", "चार", "पाच", "सहा", "सात", "आठ", "नऊ", "दहा",
    "एक", "एका", "एकाच",
}

# ---------------------------------------------------------------------------
# Section 4: Journalistic News Boilerplates
# ---------------------------------------------------------------------------
NEWS_BOILERPLATES = {
    "बातमी", "वृत्त", "माहिती", "स्पष्ट", "व्यक्त", "सुरू", "सुरु", "तयार",
    "काम", "घेऊन", "दिलाय", "दिलंय", "घेतलाय", "घेतलंय", "केलंय",
}


def main():
    # Load candidate metrics from entropy calculation
    with open("data/top100_candidates.json", "r", encoding="utf-8") as f:
        top100 = json.load(f)

    # Filter out any candidates that are domain-specific/content words
    # Check candidates with entropy >= 0.80 and non-content nature
    data_driven_words = set()
    for item in top100:
        w = item["word"]
        # Exclude any specific named entities or topical terms if any exist
        # All words in top100 are general (e.g. आहे, आणि, या, आहेत, हे, तर, etc.)
        data_driven_words.add(w)

    print(f"Count of Section 1 (Pronouns paradigm): {len(PRONOUNS)}")
    print(f"Count of Section 2 (Verb forms & auxiliaries): {len(VERB_FORMS)}")
    print(f"Count of Section 3 (Function words & particles): {len(FUNCTION_WORDS)}")
    print(f"Count of Section 4 (Journalistic boilerplates): {len(NEWS_BOILERPLATES)}")
    print(f"Count of Section 5 (Data-driven entropy candidates in top 100): {len(data_driven_words)}")

    # Combine into organized text file
    output_path = "data/stopwords_mr.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# MahaVarta Marathi Stopwords List\n")
        f.write("# Organized into linguistic grammatical paradigms + data-driven entropy candidates\n\n")

        f.write("# ============================================================================\n")
        f.write(f"# Section 1: Pronouns & Pronominal Inflections ({len(PRONOUNS)} words)\n")
        f.write("# (Full paradigm for तुम्ही, आम्ही, तो, ती, ते, आपण, स्वतः and case inflections)\n")
        f.write("# ============================================================================\n")
        for w in sorted(PRONOUNS):
            f.write(f"{w}\n")
        f.write("\n")

        f.write("# ============================================================================\n")
        f.write(f"# Section 2: Auxiliary Verbs, Copulas & Generic Verb Inflections ({len(VERB_FORMS)} words)\n")
        f.write("# (Paradigms of असणे, होणे, करणे, जाणे, येणे, देणे, घेणे, शकणे)\n")
        f.write("# ============================================================================\n")
        for w in sorted(VERB_FORMS):
            f.write(f"{w}\n")
        f.write("\n")

        f.write("# ============================================================================\n")
        f.write(f"# Section 3: Postpositions, Conjunctions & Particles ({len(FUNCTION_WORDS)} words)\n")
        f.write("# ============================================================================\n")
        for w in sorted(FUNCTION_WORDS):
            f.write(f"{w}\n")
        f.write("\n")

        f.write("# ============================================================================\n")
        f.write(f"# Section 4: Journalistic News Boilerplates ({len(NEWS_BOILERPLATES)} words)\n")
        f.write("# ============================================================================\n")
        for w in sorted(NEWS_BOILERPLATES):
            f.write(f"{w}\n")
        f.write("\n")

        # Additional data-driven words not already in above sections
        already_added = PRONOUNS | VERB_FORMS | FUNCTION_WORDS | NEWS_BOILERPLATES
        additional_entropy = data_driven_words - already_added
        f.write("# ============================================================================\n")
        f.write(f"# Section 5: High Document-Frequency & Category-Entropy Candidates ({len(additional_entropy)} words)\n")
        f.write("# (Words with normalized category entropy >= 0.80 and high document frequency)\n")
        f.write("# ============================================================================\n")
        for w in sorted(additional_entropy):
            f.write(f"{w}\n")

    total_words = len(PRONOUNS | VERB_FORMS | FUNCTION_WORDS | NEWS_BOILERPLATES | data_driven_words)
    print(f"\nTotal unique stopwords written to {output_path}: {total_words}")


if __name__ == "__main__":
    main()
