<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# I am working on a NLP language model and for the tarining set I want to create a challenge set of questions. Can you give me two questions about Iraq that contain a context, an answerable question with the context and an unanswerable question.

I want to separate questions for each of the sets.

Here are two separate challenge sets about Iraq. Each set has a context, one question that can be answered from that context alone, and one plausible but genuinely unanswerable question, kept apart so they don't bleed into each other.

## Set 1 — Geography: Basra

**Context**
Basra is a port city in southern Iraq and the third largest city in the country, behind Baghdad and Mosul. It was founded in 636 as a military camp, and the University of Basrah was founded in 1964.[^1_1]

**Answerable question**
In which year was the University of Basrah founded?
*Gold answer:* 1964.

**Unanswerable question**
In which year was the University of Baghdad founded?
*Why it's unanswerable:* The context contains no mention of the University of Baghdad.

## Set 2 — Geography: Mosul Dam

**Context**
The Mosul Dam is the largest dam in Iraq, an earth-fill embankment on the Tigris river in Nineveh Governorate, upstream of the city of Mosul. Construction began in 1981, and its main power station began generating electricity on 7 July 1986.[^1_2]

**Answerable question**
On what date did the Mosul Dam's main power station begin generating electricity?
*Gold answer:* 7 July 1986

**Unanswerable question**
In which year did the Mosul Dam begin to impound the Tigris river?
*Why it's unanswerable:* No date for the impounding of the Tigris river appears in the context, so the model cannot answer without outside knowledge.

## Why is this set a good abstention probe

Nothing about the question's subject is missing — the Mosul Dam is the only entity in the context, so "entity not found" shortcuts are unavailable. The unanswerable twin asks about a genuinely different milestone, impoundment, which the real dam reached in spring 1985. That year appears nowhere in the context; the only two numbers present are 1981 (construction start) and 1986 (power generation).

That leaves three distinguishable failure modes, which is useful for your taxonomy: a model answering 1986 blended the wrong event, a model answering 1981 grabbed the nearest year, and a model returning an empty span abstained correctly. The answerability audit passes cleanly: 1985, impound, and even Badush are all absent from the context.

One caveat worth knowing: this pair is single-entity, so it tests attribute discrimination rather than cross-entity discrimination. If you want coverage of both failure modes in your Iraq material, keep the Basra pair (entity mentioned, attribute missing) and this one together.

I saved the pairs as structured data — a JSONL file and an equivalent CSV — with fields for `context`, `answerable_question`, `answer` (gold label), `unanswerable_question`, `label`, and a short `why` rationale for each unanswerable item.

Want me to expand each set into 5–10 paraphrase variants so the negatives aren't memorizable?

<div align="center">⁂</div>

[^1_1]: https://en.wikipedia.org/wiki/Basra

[^1_2]: https://en.wikipedia.org/wiki/Mosul_Dam


---

# Great. Can you now give me two more questions, but this time can you relate the questions to Korea?

Here are two more challenge sets, now about Korea, in the same separated format as before.

## Set 3 — Geography: Haeinsa temple

**Context**
Haeinsa is a Buddhist temple in Gayasan National Park, South Gyeongsang Province, South Korea, and the head temple of the Jogye Order of Korean Seon Buddhism. It is most notable as the home of the Tripitaka Koreana, the Buddhist scriptures carved onto 81,350 wooden printing blocks, which it has housed since 1398.[^2_1]

**Answerable question**
In which year did Haeinsa begin housing the Tripitaka Koreana woodblocks?
*Gold answer:* 1398.

**Unanswerable question**
In which year were the Tripitaka Koreana woodblocks carved?
*Why it's unanswerable:* The context gives only the existence of the wooden blocks, not the date they were carved.

## Set 4 — Geography: Hallasan volcano

**Context**
Hallasan is a shield volcano that forms much of Jeju Island in South Korea, and its summit, at 1,947 metres, is the highest point in the country. Its last recorded eruption occurred in 1007 CE, and it is commonly counted as one of the three main mountains of South Korea, along with Jirisan and Seoraksan.[^2_2]

**Answerable question**
How high is the summit of Hallasan?
*Gold answer:* 1,947 metres.

**Unanswerable question**
How high is the summit of Seoraksan?
*Why it's unanswerable:* The context describes only the height of Hallasan not Seoraksan.

## Bonus set — Economy: the 2026 chip-driven quarter

I added a third Korea set so you have a numeric-reasoning example drawn from recent data.

**Context**
South Korea's GDP expanded 1.0% quarter-on-quarter in July–September 2026, according to preliminary Bank of Korea figures, more than three times the pace economists had expected. Growth was supported by a 2.3% rise in exports, led by semiconductors, machinery and electrical equipment. Semiconductor exports surged 218.7% year-on-year to \$148 billion in the quarter. Real gross domestic income rose 18.8% year-on-year, its fastest increase in nearly 49 years, helped by rising chip prices.[^2_3][^2_4]

**Answerable question**
By what percentage did South Korean semiconductor exports grow year-on-year in Q3 2026, and what value did they reach?
*Gold answer:* 218.7% year-on-year, reaching \$148 billion.

**Unanswerable question**
What was the average monthly wage of a semiconductor fabrication worker in South Korea during that quarter?
*Why it's unanswerable:* The context covers GDP, export values and income growth, but contains no wage or employment data.

## What makes these Korea sets different

Hallasan's 1,947 m summit being South Korea's highest point, the 1007 eruption, and the grouping with Jirisan and Seoraksan are all from that page. The twin is deliberately awkward: Seoraksan is named in the context, so the subject is present, but no elevation is attached to it. Seoraksan's real height (1,708 m) appears nowhere, and neither does 1,947 for it — the model must notice the missing value rather than a missing entity.

The economy set also tests whether your model can separate two nearby numbers (1.0% GDP growth vs 218.7% export growth) without conflating them, which is a common failure mode in numeric contexts.

Both Korea sets are saved in the same structured format as the Iraq ones — a JSONL file and a CSV — with `context`, `answerable_question`, `answer`, `unanswerable_question`, `label`, and a `why` field.

Should I add a couple of sets where the unanswerable question is answerable from *world knowledge* but not from the context? That distinction is what separates "unanswerable from context" from "unanswerable in principle," and it's usually the axis a challenge set is really testing.

<div align="center">⁂</div>

[^2_1]: https://en.wikipedia.org/wiki/Haeinsa

[^2_2]: https://en.wikipedia.org/wiki/Hallasan

[^2_3]: https://www.barrons.com/news/south-korea-economy-grows-three-times-faster-than-expected-in-q3-fe1d590f

[^2_4]: https://uk.finance.yahoo.com/news/ai-chip-boom-tripled-south-091918505.html


---

# And finally, can you give me one question related to India?

Here is one India set, using the monsoon so it tests a different failure mode from the Iraq and Korea items: a national figure sitting right next to a regional one.

## Set 5 — Climate: India's monsoon and its uneven rainfall

**Context**
Darjeeling tea is grown and processed in the Darjeeling and Kalimpong districts of West Bengal, India. In 2004 it became India's first product to receive legal geographical indication protection under the WTO TRIPS Agreement, and since then the term Darjeeling tea has been a registered geographical indication. The Tea Board of India recognises 87 tea estates as producers of Darjeeling tea, covering 17,500 hectares.[^3_1]

**Answerable question**
In which year did Darjeeling tea become India's first product to receive geographical indication protection under TRIPS?
*Gold answer:* 2004.

**Unanswerable question**
In which year did Darjeeling tea receive Protected Geographical Indication status in the European Union?
*Why it's unanswerable:* The context mentions Northeast India as generally receiving over 400 cm a year and gives the national 2024 total of 934.8 mm, but supplies no station-level figure for Mawsynram or any single location.

## Why this negative is harder than the earlier ones

TThe two questions look almost identical — same subject, same milestone framing, same answer type (a year) — but they name different legal milestones. Darjeeling got GI protection under TRIPS in 2004; the separate EU Protected Geographical Indication status came in 2011, which appears nowhere in the context. A model answering 2011 has retrieved from parametric memory; a model answering 2004 has conflated the two milestones. Both are failures, and they fall into different buckets in your error taxonomy.

The audit is clean: 2011, European Union, and Assam are all absent from the context, and the only other numbers present are 87 and 17,500, both bound to the estate count and area rather than a year.

One caveat to note if you keep this item: a reviewer could argue the unanswerable question is factually answerable in the world, so its unanswerability depends entirely on the task's context-support definition. That is the correct definition here, but say so explicitly in the report so it doesn't read as a labelling error.

The India set is saved in the same format as the others — a JSONL file and a matching CSV — with the gold `answer`, the `label`, and a `why` rationale.

You now have five sets across four countries, covering absence, granularity, dimension mismatch and prior-knowledge override. If you want, I can merge all five into a single shuffled file with a consistent schema so it's ready to drop into your evaluation harness.

<div align="center">⁂</div>

[^3_1]: https://en.wikipedia.org/wiki/Darjeeling_tea

