# server.py
# FastAPI backend for POS / Phrases / Clauses trainer

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import random
import requests

# ---------- CORS (IMPORTANT) ----------
# Replace YOUR-USERNAME with your GitHub username (no trailing slash)
GITHUB_PAGES_ORIGIN = "https://YOUR-USERNAME.github.io"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[GITHUB_PAGES_ORIGIN],  # e.g. "https://omarh.github.io"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- spaCy setup (auto-download model on first run) ----------
import spacy
from spacy.util import is_package
from spacy.cli import download as spacy_download

MODEL_NAME = "en_core_web_sm"
if not is_package(MODEL_NAME):
    spacy_download(MODEL_NAME)

nlp = spacy.load(MODEL_NAME, disable=[])  # enable everything small model has

# ---------- Local fallback sentences ----------
LOCAL_SENTENCES = [
    "Wow! That absolutely crushed the previous record.",
    "She checked out of the hotel early this morning.",
    "The committee has been reviewing the proposal for weeks.",
    "Those quickly written notes were surprisingly clear.",
    "After the storm, the lights finally came back on.",
    "I will look into the issue tomorrow.",
    "The red car in the driveway belongs to my neighbor.",
    "Because it was raining, we decided to stay inside.",
    "Please hand in your assignments by Friday.",
    "They might have been waiting for the bus."
]

# ---------- Simple helpers for analysis ----------
LINKING_LEMMAS = {"be", "seem", "become", "appear", "feel", "look", "sound", "remain", "stay", "grow", "turn", "smell", "taste"}
MODALS = {"can","could","may","might","must","shall","should","will","would"}
PARTICLE_POS = {"PART", "ADP", "ADV"}  # for phrasal verbs (spaCy tags particles as PART; sometimes ADP/ADV)

def is_aux(tok) -> bool:
    # Auxiliaries include be/have/do (as AUX) and modals (tag 'MD')
    return tok.pos_ == "AUX" or tok.tag_ == "MD" or tok.lemma_ in ("be","have","do")

def aux_role(aux_tok, head_tok) -> str:
    if aux_tok.tag_ == "MD" or aux_tok.lemma_ in MODALS:
        return "modal"
    # perfect: have + VBN
    if aux_tok.lemma_ == "have":
        return "perfect"
    # progressive/passive: be + VBG/VBN (we'll guess by head tag)
    if aux_tok.lemma_ == "be":
        if head_tok.tag_ == "VBG":
            return "progressive"
        if head_tok.tag_ == "VBN":
            return "passive"
    # do-support
    if aux_tok.lemma_ == "do":
        return "do-support"
    return "other"

def collect_aux_chain(tok) -> List[str]:
    auxes = [c for c in tok.children if c.dep_ in ("aux","auxpass")]
    # also include preceding MD tokens
    md_before = []
    for t in tok.doc:
        if t.head == tok and t.tag_ == "MD":
            md_before.append(t)
    auxes.extend(md_before)
    # Sort by token index
    auxes = sorted(set(auxes), key=lambda x: x.i)
    return [a.text for a in auxes]

def find_phrasal_particles(verb_tok) -> List[str]:
    """Return particles that form phrasal verb with the verb (e.g., 'check out')."""
    parts = []
    for c in verb_tok.children:
        # particles are usually 'prt' dependency in some models, but en_core_web_sm uses 'prt' rarely.
        # We'll include ADP/ADV/PART that attach with dep_ in ("prt","prep","advmod") and are adjacent/rightward.
        if c.dep_ in ("prt", "prep", "advmod") and c.pos_ in PARTICLE_POS and c.i > verb_tok.i:
            parts.append(c.text)
    return parts

def has_object(verb_tok) -> bool:
    """Heuristic: transitive if it has a direct object or clausal complement."""
    for c in verb_tok.children:
        if c.dep_ in ("dobj","obj","attr","ccomp","xcomp","oprd"):
            return True
    # passive subject as object proxy
    for c in verb_tok.children:
        if c.dep_ in ("nsubjpass",):
            return True
    return False

def noun_info(tok) -> Dict[str, Any]:
    txt = tok.text
    is_proper = tok.tag_ in ("NNP","NNPS")
    is_plural = tok.tag_ in ("NNS","NNPS")
    # possessive if head is possessive marker or token has 'poss' child/head pattern
    possessive = any(c.dep_=="case" and c.text=="'s" for c in tok.children) or (tok.dep_ == "poss")
    concreteness = "concrete" if tok.ent_type_ or tok.tag_.startswith("NN") else "abstract"
    countability = "count" if is_plural or tok.tag_ in ("NN","NNS","NNP","NNPS") else "noncount"
    # naive collective list
    collective_examples = {"team","committee","group","audience","family","staff","class","crew","troop","jury"}
    collective = tok.lemma_.lower() in collective_examples
    why = []
    if is_proper: why.append("Proper: capitalized name or title form.")
    else: why.append("Common: general category word.")
    if possessive: why.append("Possessive form detected.")
    if collective: why.append("Collective noun (group as a unit).")
    return {
        "i": tok.i, "text": txt, "lemma": tok.lemma_,
        "type": "proper" if is_proper else "common",
        "concreteness": concreteness,
        "countability": countability,
        "collective": collective,
        "possessive": possessive,
        "why": why or ["General noun usage in context."]
    }

def pronoun_info(tok) -> Dict[str, Any]:
    txt = tok.text.lower()
    # classify pronoun types roughly
    personal_subj = {"i","we","you","he","she","they","it"}
    personal_obj = {"me","us","you","him","her","them","it"}
    possessive_abs = {"mine","yours","his","hers","ours","theirs"}
    possessive_dep = {"my","your","his","her","our","their","its"}
    demonstratives = {"this","that","these","those"}
    interrogatives = {"who","whom","whose","which","what"}
    relatives = {"who","whom","whose","which","that"}
    reflexive = {"myself","yourself","himself","herself","itself","ourselves","yourselves","themselves"}
    reciprocal = {"each other","one another"}  # multi-word not handled perfectly

    ptype = "other"
    case = "(n/a)"
    poss_form = None
    refl = False
    why = []

    if txt in personal_subj:
        ptype = "personal"; case = "subjective"; why.append("Personal pronoun (subjective case).")
    elif txt in personal_obj:
        ptype = "personal"; case = "objective"; why.append("Personal pronoun (objective case).")
    elif txt in possessive_abs:
        ptype = "possessive"; poss_form = "absolute"; why.append("Possessive pronoun (absolute).")
    elif txt in possessive_dep:
        ptype = "possessive"; poss_form = "dependent"; why.append("Possessive determiner (dependent).")
    elif txt in demonstratives:
        ptype = "demonstrative"; why.append("Demonstrative pronoun.")
    elif txt in interrogatives and tok.dep_ in ("attr","pobj","dobj","nsubj"):
        ptype = "interrogative"; why.append("Interrogative pronoun.")
    elif txt in relatives and tok.dep_ in ("relcl","nsubj","dobj","pobj","attr","relcl"):
        ptype = "relative"; why.append("Relative pronoun within a clause.")
    elif txt in reflexive:
        ptype = "reflexive"; refl = True; why.append("Reflexive/Intensive pronoun.")
    else:
        ptype = "indefinite" if tok.tag_ == "PRP$" or tok.tag_ == "PRP" else "other"

    return {
        "i": tok.i, "text": tok.text, "lemma": tok.lemma_, "type": ptype,
        "case": case, "possessive_form": poss_form, "reflexive_or_intensive": refl,
        "why": why or ["Pronoun identified by POS tag and context."]
    }

def adjective_info(tok) -> Dict[str, Any]:
    deg = "positive"
    if tok.tag_ in ("JJR",): deg = "comparative"
    if tok.tag_ in ("JJS",): deg = "superlative"
    typ = "proper" if tok.text[:1].isupper() and tok.dep_ in ("amod","acomp") else "common"
    return {
        "i": tok.i, "text": tok.text, "lemma": tok.lemma_,
        "type": typ, "degree": deg,
        "why": [f"Adjective ({tok.tag_}); degree: {deg}."]
    }

def adverb_info(tok) -> Dict[str, Any]:
    deg = "positive"
    if tok.tag_ == "RBR": deg = "comparative"
    if tok.tag_ == "RBS": deg = "superlative"
    conj_adv = tok.lemma_.lower() in {"however","therefore","moreover","consequently","nevertheless","furthermore","thus","hence","meanwhile"}
    modifies = "verb"
    if tok.dep_ == "advmod" and tok.head.pos_ == "ADJ": modifies = "adjective"
    elif tok.dep_ == "advmod" and tok.head.pos_ == "ADV": modifies = "adverb"
    elif tok.dep_ == "advmod" and tok.head.pos_ in ("VERB","AUX"): modifies = "verb"
    elif tok.dep_ == "advcl": modifies = "sentence"
    why = [f"Adverb ({tok.tag_}); degree: {deg}."]
    if conj_adv: why.append("Conjunctive adverb (links clauses).")
    return {
        "i": tok.i, "text": tok.text, "lemma": tok.lemma_,
        "modifies": modifies, "degree": deg, "conjunctive": conj_adv,
        "why": why
    }

def preposition_info(tok) -> Dict[str, Any]:
    return {"i": tok.i, "text": tok.text, "lemma": tok.lemma_, "type": "preposition", "why": ["Preposition heads a PP or marks relation."]}

def conjunction_info(tok) -> Dict[str, Any]:
    lemma = tok.lemma_.lower()
    if lemma in {"for","and","nor","but","or","yet","so"}:
        typ = "coordinating"
        why = ["Coordinating conjunction (FANBOYS)."]
    elif lemma in {"because","although","if","when","since","while","after","before","though","unless","until"}:
        typ = "subordinating"
        why = ["Subordinating conjunction (introduces dependent clause)."]
    elif lemma in {"however","therefore","moreover","consequently","furthermore","nevertheless"}:
        typ = "conjunctive_adverb"
        why = ["Conjunctive adverb connecting clauses."]
    else:
        typ = "other"; why = ["Conjunction."]
    return {"i": tok.i, "text": tok.text, "lemma": tok.lemma_, "type": typ, "why": why}

def interjection_info(tok) -> Dict[str, Any]:
    return {"i": tok.i, "text": tok.text, "lemma": tok.lemma_, "is_interjection": True, "why": ["Interjection expresses emotion/reaction."]}

# ---------- Endpoints ----------

class AnalyzeOut(BaseModel):
    text: str
    tokens: List[Dict[str, Any]]
    verbs: List[Dict[str, Any]]
    auxiliaries: List[Dict[str, Any]]
    nouns: List[Dict[str, Any]]
    pronouns: List[Dict[str, Any]]
    adjectives: List[Dict[str, Any]]
    adverbs: List[Dict[str, Any]]
    prepositions: List[Dict[str, Any]]
    conjunctions: List[Dict[str, Any]]
    interjections: List[Dict[str, Any]]
    noun_phrases: List[Dict[str, Any]]
    clauses: List[Dict[str, Any]]
    pos_labels: List[Dict[str, Any]]

@app.get("/sentence")
def sentence(source: str = Query("local", enum=["local","tatoeba","wordnik"])) -> Dict[str, Any]:
    """
    Returns a sentence. Defaults to local fallback so the site works
    even if external APIs are blocked.
    """
    if source == "tatoeba":
        # tiny public endpoint with random sentences
        try:
            r = requests.get("https://tatoeba.org/en/api_v0/search?query=&from=eng&to=eng&sort=random&limit=1", timeout=6)
            j = r.json()
            if j.get("results"):
                txt = j["results"][0]["text"]
                return {"text": txt, "source": "tatoeba"}
        except Exception:
            pass
    if source == "wordnik":
        key = os.environ.get("WORDNIK_API_KEY")
        if key:
            try:
                r = requests.get(f"https://api.wordnik.com/v4/words.json/randomWord?hasDictionaryDef=true&api_key={key}", timeout=6)
                base = r.json().get("word")
                if base:
                    # get an example sentence
                    ex = requests.get(f"https://api.wordnik.com/v4/word.json/{base}/examples?api_key={key}", timeout=6).json()
                    examples = ex.get("examples") or []
                    if examples:
                        txt = examples[0].get("text") or base
                        return {"text": txt, "source": "wordnik"}
            except Exception:
                pass
    # local fallback
    return {"text": random.choice(LOCAL_SENTENCES), "source": "local"}

@app.get("/analyze", response_model=AnalyzeOut)
def analyze(text: str):
    """
    Analyze sentence: tokens, verbs (with phrasal particles, transitivity, linking),
    auxiliaries, POS buckets, noun phrases, simple clause info, and labels+why for full game.
    """
    doc = nlp(text)

    tokens = [{"i": t.i, "text": t.text, "lemma": t.lemma_, "pos": t.pos_, "tag": t.tag_, "dep": t.dep_} for t in doc]

    # POS buckets
    verbs, auxiliaries, nouns, pronouns, adjectives, adverbs = [], [], [], [], [], []
    prepositions, conjunctions, interjections = [], [], []

    # Full POS labels with "why"
    pos_labels = []

    for t in doc:
        # Aux first
        if is_aux(t):
            role = aux_role(t, t.head if t.head else t)
            auxiliaries.append({
                "i": t.i, "text": t.text, "lemma": t.lemma_,
                "pos": t.pos_, "tag": t.tag_, "role": role,
                "head_text": t.head.text if t.head else None,
                "why": [f"Auxiliary ({role})."]
            })
            pos_labels.append({"i": t.i, "gold": "auxiliary", "why": [f"Auxiliary '{t.text}' ({role})."]})
            continue

        if t.pos_ == "VERB":
            particles = find_phrasal_particles(t)
            phrasal = (" ".join([t.text] + particles)) if particles else t.text
            trans = "transitive" if has_object(t) else "intransitive"
            is_link = (t.lemma_ in LINKING_LEMMAS) and (t.dep_ in ("ROOT","xcomp","ccomp","acomp","attr","relcl","conj"))
            soa = "state" if is_link else ("occurrence" if t.lemma_ in {"happen","occur","arise"} else "action")
            verbs.append({
                "i": t.i, "text": t.text, "lemma": t.lemma_, "pos": t.pos_, "tag": t.tag_,
                "aux_chain": collect_aux_chain(t),
                "particles": particles, "phrasal": phrasal,
                "transitivity": trans, "is_linking": bool(is_link), "is_modal": False,
                "soa": soa,
                "why": [
                    f"Verb detected ({t.tag_}).",
                    f"Phrasal particles: {' '.join(particles) if particles else 'none'}.",
                    f"Transitivity: {trans}.",
                    ("Linking verb (state)" if is_link else "Action/occurrence based on context.")
                ]
            })
            pos_labels.append({"i": t.i, "gold": "verb", "why": [f"Main verb '{t.text}'."]})
            continue

        if t.pos_ == "NOUN" or t.tag_ in ("NN","NNS","NNP","NNPS"):
            info = noun_info(t)
            nouns.append(info)
            pos_labels.append({"i": t.i, "gold": "noun", "why": info["why"]})
            continue

        if t.pos_ == "PRON":
            info = pronoun_info(t)
            pronouns.append(info)
            pos_labels.append({"i": t.i, "gold": "pronoun", "why": info["why"]})
            continue

        if t.pos_ == "ADJ":
            info = adjective_info(t)
            adjectives.append(info)
            pos_labels.append({"i": t.i, "gold": "adjective", "why": info["why"]})
            continue

        if t.pos_ == "ADV":
            info = adverb_info(t)
            adverbs.append(info)
            pos_labels.append({"i": t.i, "gold": "adverb", "why": info["why"]})
            continue

        if t.pos_ == "ADP":
            info = preposition_info(t)
            prepositions.append(info)
            pos_labels.append({"i": t.i, "gold": "preposition", "why": info["why"]})
            continue

        if t.pos_ == "CCONJ" or t.pos_ == "SCONJ":
            info = conjunction_info(t)
            conjunctions.append(info)
            pos_labels.append({"i": t.i, "gold": "conjunction", "why": info["why"]})
            continue

        if t.pos_ == "INTJ":
            info = interjection_info(t)
            interjections.append(info)
            pos_labels.append({"i": t.i, "gold": "interjection", "why": info["why"]})
            continue

        # everything else
        pos_labels.append({"i": t.i, "gold": "other", "why": ["Not a core POS in this game."]})

    # Noun phrases (spaCy noun_chunks)
    noun_phrases = []
    if doc.has_annotation("DEP"):
        for np in doc.noun_chunks:
            head = np.root
            has_det = any(t.dep_ == "det" for t in np)
            pp_post = any(t.dep_ == "prep" for t in np)
            head_type = "proper" if head.tag_ in ("NNP","NNPS") else "common"
            # rough role guess
            role = "subject" if head.dep_ in ("nsubj","nsubjpass") else ("object" if head.dep_ in ("dobj","obj") else ("object_of_preposition" if head.dep_=="pobj" else "other"))
            noun_phrases.append({
                "i": head.i,
                "head": head.text,
                "span": np.text,
                "role": role,
                "has_det": has_det,
                "pp_postmod": pp_post,
                "head_type": head_type,
                "why": [f"Head '{head.text}' with NP span '{np.text}'."]
            })

    # Clauses (very lightweight: treat ROOT and conjuncts / ccomp/xcomp as clause-like)
    clauses = []
    for t in doc:
        if t.dep_ == "ROOT" and t.pos_ in ("VERB","AUX"):
            span = t.subtree
            text_span = " ".join(tok.text for tok in span)
            clauses.append({
                "i": t.i,
                "text": text_span,
                "type": "independent",
                "finite": True,
                "has_marker": False,
                "why": ["Root predicate => independent clause."]
            })
        if t.dep_ in ("ccomp","xcomp","advcl","relcl"):
            span = t.subtree
            text_span = " ".join(tok.text for tok in span)
            cl_type = "adverbial" if t.dep_=="advcl" else ("relative" if t.dep_=="relcl" else "complement")
            has_marker = any(ch.dep_ in ("mark","complm","relcl") for ch in t.children)
            clauses.append({
                "i": t.i,
                "text": text_span,
                "type": cl_type,
                "finite": t.morph.get("VerbForm") != ["Inf"],
                "has_marker": has_marker,
                "why": [f"{t.dep_} attached to {t.head.text}."]
            })

    out = {
        "text": text,
        "tokens": tokens,
        "verbs": verbs,
        "auxiliaries": auxiliaries,
        "nouns": nouns,
        "pronouns": pronouns,
        "adjectives": adjectives,
        "adverbs": adverbs,
        "prepositions": prepositions,
        "conjunctions": conjunctions,
        "interjections": interjections,
        "noun_phrases": noun_phrases,
        "clauses": clauses,
        "pos_labels": pos_labels,
    }
    return out
