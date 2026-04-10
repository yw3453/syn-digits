# Data Documentation

This document describes each file in the `data/` folder, including its purpose, structure, and examples.

## `personas.json`

**Purpose:** Contains textual persona descriptions for 2,058 synthetic individuals. These descriptions are used as prompts to condition LLMs when simulating survey responses or movie ratings.

**Structure:** A JSON object mapping persona IDs (`pid_<number>`) to multi-line text strings. Each persona description includes:

- Demographic attributes (region, gender, age, education, race, citizenship, marital status, religion, religious attendance, political affiliation, income, political views, household size, employment status)
- Big 5 personality scores with percentile rankings

**Example (truncated):**

```json
{
  "pid_574": "The following is a description of a person.\n\nThe person's demographics are the following...\nGeographic region: South (TX, OK, AR, LA, ...)\nGender: Male\nAge: 18-29\nEducation level: Some college, no degree\nRace: White\n...\n\nThe person's Big 5 scores are the following:\nscore_extraversion = 3.5 (75th percentile)\nscore_agreeableness = 4.111 (62nd percentile)\n...",
  "pid_2001": "..."
}
```

---

## `2K500/synthetic_control/`

These files are used in the **synthetic control** task for the Twin-2K-500 dataset. The directory contains multiple subdirectories, one per persona construction (persona format × LLM model combination):

```
2K500/synthetic_control/
├── Text Persona - GPT4.1-mini/
├── JSON Persona - GPT4.1/
├── JSON Persona - GPT4.1-mini/
├── JSON Persona (Predicted Output) - GPT4.1/
├── JSON Persona (Predicted Output) - GPT4.1-mini/
├── Text Persona (Default Temperature) - GPT4.1-mini/
├── Text Persona (Reasoning) - GPT4.1-mini/
├── Text Persona (Repeating Questions) - GPT4.1-mini/
├── Text Persona - Gemini-Flash2.5/
├── LLM Finetuning (500 training samples) - GPT4.1-mini/
├── Demographics Only - GPT4.1-mini/
├── Persona Summary - GPT4.1-mini/
└── Persona Summary - JSON Persona - GPT4.1-mini/
```

Each subdirectory contains a `real.csv` and `LLM.csv` pair.

### `<persona_construction>/real.csv`

**Purpose:** Contains real survey responses from human participants.

**Structure:** CSV with rows (one per respondent) and 124 columns. The first column `TWIN_ID` is the respondent identifier; the remaining 123 columns are survey question responses. Values are numeric (integers and floats), with missing values represented as empty cells. The number of rows varies by persona construction (most have 2,058; some have fewer due to filtering).

**Example (first 3 rows, selected columns):**

```
TWIN_ID,False Cons. self _1,False Cons. self _2,...,1_Q295,2_Q295,...
1634,1,1,...,1,2,...
1747,1,1,...,2,2,...
```

### `<persona_construction>/LLM.csv`

**Purpose:** Contains LLM-generated (simulated) survey responses for the same respondents and questions as `real.csv`. Used as a baseline to compare against the calibrated synthetic control method.

**Structure:** Identical schema to `real.csv` — same rows and columns with the same `TWIN_ID` identifiers and question columns.

---

## `MovieLens-20M/synthetic_control/`

These files are used in the **synthetic control** task for the MovieLens-20M dataset (movie ratings with 500 users and 250 movies).

### `real.csv`

**Purpose:** Contains real movie ratings from human users.

**Structure:** CSV with 500 rows (one per user) and 251 columns. The first column `userId` is the user identifier; the remaining 250 columns are movie IDs. Values are ratings on a 0.5–5.0 scale (in 0.5 increments), with missing values for unrated movies.

**Example (first 2 rows, selected columns):**

```
userId,2,6,16,22,25,...
156,5.0,4.0,4.0,4.0,4.0,...
741,3.0,3.5,4.0,2.5,4.5,...
```

### `LLM.csv`

**Purpose:** Contains LLM-generated (simulated) movie ratings for the same users and movies as `real.csv`.

**Structure:** Identical schema to `real.csv` — 500 rows x 251 columns with the same `userId` identifiers and movie ID columns. Ratings are on the same 0.5–5.0 scale.

### `LLM_in_context_other_ratings_full.csv`

**Purpose:** Contains LLM-generated movie ratings where each prediction is conditioned on the user's ratings for all other movies provided as in-context examples. Used as an additional baseline in `MovieLens_experiments.ipynb` to compare against the synthetic control method.

**Structure:** Identical schema to `real.csv` — 500 rows x 251 columns with the same `userId` identifiers and movie ID columns. Ratings are on the same 0.5–5.0 scale.

---

## `MovieLens-20M/distribution_calibration/`

These files are used in the **distribution calibration** task for the MovieLens-20M dataset.

### `top_movies_df.csv`

**Purpose:** Contains metadata and rating distributions for the 500 most-rated movies in the MovieLens-20M dataset. Used to define target marginal distributions that the calibration algorithm matches against.

**Structure:** CSV with 500 rows (one per movie) and 14 columns:

| Column | Description |
|--------|-------------|
| `title` | Movie title and release year |
| `genres` | Pipe-separated genre list |
| `top_10_tags` | Comma-separated list of the 10 most popular user-applied tags |
| `avg_rating` | Average rating across all users |
| `0.5` through `5.0` | Count of ratings at each half-star increment (10 columns) |

**Example (first 2 rows):**

```
title,genres,top_10_tags,avg_rating,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0
"Toy Story (1995)",Adventure|Animation|Children|Comedy|Fantasy,"['toys', 'computer animation', ...]",3.921,178,506,266,1440,1060,8751,4200,17136,3890,12268
"Jumanji (1995)",Adventure|Children|Fantasy,"['adventure', 'jungle', ...]",3.212,211,708,429,2227,1532,7598,2266,5156,452,1664
```

### `persona_ratings.csv`

**Purpose:** Contains LLM-generated movie ratings from each persona for all 500 movies. These are the raw simulated ratings before calibration.

**Structure:** CSV with 500 rows (one per movie) and 2,060 columns. The first column is the index (a numeric movie identifier), followed by `avg_rating` (the ground-truth average rating for that movie), followed by 2,058 persona columns (named `pid_<number>`). Ratings are on the 0.5–5.0 scale.

**Example (first 2 rows, selected columns):**

```
,avg_rating,pid_574,pid_2001,pid_1710,...
1,3.921,4.5,4.5,3.0,...
2,3.212,4.0,4.0,3.0,...
```

Here the first (unnamed) column is the movie index, `avg_rating` is the ground-truth average, and each `pid_*` column holds that persona's rating.

---

## `OpinionQA/distribution_calibration/`

These files are used in the **distribution calibration** task for the OpinionQA dataset (opinion survey questions from Pew Research).

### `Qs_likert_scale_5_choices.json`

**Purpose:** Contains survey questions with 5-choice Likert-scale answer options, along with the ground-truth answer distributions from Pew Research surveys. These define the target marginal distributions for calibration.

**Structure:** A JSON object mapping question IDs (e.g., `SAFECRIME_W26`) to objects with:

| Field | Description |
|-------|-------------|
| `question` | Full question text with answer options |
| `choices` | Ordered list of answer choice labels |
| `choice_to_numeric` | Mapping from choice labels to numeric codes |
| `choice_counts` | Distribution of responses across choices (keyed by numeric code) |

**Example:**

```json
{
  "SAFECRIME_W26": {
    "question": "How safe, if at all, would you say your local community is from crime? Would you say it is ['1. Very safe', '2. Somewhat safe', '3. Not too safe', '4. Not at all safe', '5. Refused']",
    "choices": ["Very safe", "Somewhat safe", "Not too safe", "Not at all safe", "Refused"],
    "choice_to_numeric": {
      "Very safe": 1,
      "Somewhat safe": 2,
      "Not too safe": 3,
      "Not at all safe": 4,
      "Refused": 5
    },
    "choice_counts": {
      "1": 1169,
      "2": 2381,
      "3": 491,
      "4": 109,
      "5": 10
    }
  }
}
```

There are **489 questions** in total.

### `persona_answers.csv`

**Purpose:** Contains LLM-generated survey answers from each persona for all 489 questions. These are the raw simulated answers before calibration.

**Structure:** CSV with 489 rows (one per question, indexed by question ID) and 2,058 columns (one per persona, named `pid_<number>`). Values are integers corresponding to the numeric answer codes defined in `Qs_likert_scale_5_choices.json`.

**Example (first 2 rows, selected columns):**

```
,pid_574,pid_2001,pid_1710,pid_1277,...
SAFECRIME_W26,2,2,2,3,...
GUNIDENTITY_W26,3,4,4,4,...
```
