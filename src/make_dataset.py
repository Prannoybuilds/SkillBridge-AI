"""
make_dataset.py
----------------
Generates a synthetic Resume <-> Job-Description matching dataset.

Why synthetic data?
Real resume/JD datasets are usually large scraped corpora that come with
licensing/privacy baggage. For a learning project, a synthetic generator
gives us full control over signal (skill overlap, seniority, domain) so the
model has something genuine to learn, while keeping the whole pipeline
reproducible and lightweight.

Output: data/train.jsonl
Each line: {"resume": str, "job_description": str, "label": 0/1}
label = 1  -> resume is a good match for the JD
label = 0  -> resume is a poor match for the JD (wrong domain / missing skills)
"""

import json
import random

random.seed(42)

DOMAINS = {
    "data_science": {
        "skills": ["python", "pandas", "numpy", "scikit-learn", "tensorflow",
                    "pytorch", "sql", "statistics", "machine learning",
                    "deep learning", "data visualization", "nlp"],
        "titles": ["Data Scientist", "Machine Learning Engineer", "ML Researcher"],
    },
    "frontend": {
        "skills": ["javascript", "react", "html", "css", "typescript",
                    "redux", "webpack", "next.js", "responsive design", "figma"],
        "titles": ["Frontend Developer", "UI Engineer", "React Developer"],
    },
    "backend": {
        "skills": ["java", "spring boot", "microservices", "rest api",
                    "docker", "kubernetes", "postgresql", "kafka", "aws", "system design"],
        "titles": ["Backend Developer", "Software Engineer", "API Developer"],
    },
    "devops": {
        "skills": ["ci/cd", "jenkins", "terraform", "aws", "azure",
                    "kubernetes", "docker", "monitoring", "linux", "ansible"],
        "titles": ["DevOps Engineer", "Site Reliability Engineer", "Cloud Engineer"],
    },
    "marketing": {
        "skills": ["seo", "content strategy", "google analytics", "social media",
                    "email marketing", "brand strategy", "campaign management", "copywriting"],
        "titles": ["Marketing Manager", "Growth Marketer", "Content Strategist"],
    },
}

SENIORITY = ["Fresher", "1-2 years experience", "3-5 years experience", "5+ years experience"]

RESUME_TEMPLATE = (
    "{name} is a {seniority} professional. Core skills include {skills}. "
    "Previously worked on projects involving {project_focus}. "
    "Looking for a role as {title}."
)

JD_TEMPLATE = (
    "We are hiring a {title} with {seniority}. "
    "Required skills: {skills}. "
    "The candidate will work on {project_focus} in a collaborative team environment."
)

NAMES = ["Candidate A", "Candidate B", "Candidate C", "Candidate D", "Candidate E"]


def sample_skills(domain, k=6):
    pool = DOMAINS[domain]["skills"]
    k = min(k, len(pool))
    return random.sample(pool, k)


def make_resume(domain):
    skills = sample_skills(domain, k=random.randint(4, 7))
    title = random.choice(DOMAINS[domain]["titles"])
    text = RESUME_TEMPLATE.format(
        name=random.choice(NAMES),
        seniority=random.choice(SENIORITY),
        skills=", ".join(skills),
        project_focus=f"{domain.replace('_', ' ')} systems",
        title=title,
    )
    return text, set(skills)


def make_jd(domain):
    skills = sample_skills(domain, k=random.randint(4, 7))
    title = random.choice(DOMAINS[domain]["titles"])
    text = JD_TEMPLATE.format(
        title=title,
        seniority=random.choice(SENIORITY),
        skills=", ".join(skills),
        project_focus=f"{domain.replace('_', ' ')} systems",
    )
    return text, set(skills)


def build_examples(n_per_domain=120):
    examples = []
    domains = list(DOMAINS.keys())

    for domain in domains:
        for _ in range(n_per_domain):
            # Positive pair: resume and JD from the SAME domain (should match)
            resume_text, _ = make_resume(domain)
            jd_text, _ = make_jd(domain)
            examples.append({"resume": resume_text, "job_description": jd_text, "label": 1})

            # Negative pair: resume from this domain, JD from a DIFFERENT domain
            other_domain = random.choice([d for d in domains if d != domain])
            jd_text_neg, _ = make_jd(other_domain)
            examples.append({"resume": resume_text, "job_description": jd_text_neg, "label": 0})

    random.shuffle(examples)
    return examples


if __name__ == "__main__":
    data = build_examples(n_per_domain=90)
    out_path = "data/train.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")

    n_pos = sum(1 for r in data if r["label"] == 1)
    print(f"Wrote {len(data)} examples to {out_path}")
    print(f"Positive (match): {n_pos}  Negative (no match): {len(data) - n_pos}")
