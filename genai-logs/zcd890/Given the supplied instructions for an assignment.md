<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Given the supplied instructions for an assignment and my individual assignment report, can you write me a contribution statement?

I've read both files. The instructions require the contribution statement to (a) state which core components were joint versus individual, (b) name who holds primary responsibility for each extension, and (c) sit last in the report, after the conclusion, and outside the page limit. Your report is titled "Individual Report" with a single author, so I've drafted it as a solo statement — with a fallback paragraph if you were actually part of a group.[^1_1]

## Ready-to-paste statement

**Contribution Statement**

This project was carried out individually by Kieran O'Neill (zcd890). I had no group members, so every core component (Weeks 36–41), the paired challenge set, and the individual extension were designed, implemented, evaluated and written by me alone; nothing was completed jointly. Consequently I take full responsibility for all parts of the submitted report, code and challenge set.


| Component | Responsibility | What I did |
| :-- | :-- | :-- |
| W36 data inspection (Items 1–3) | Individual | Per-language and per-split example counts, answerability proportions and length statistics; top-5 question tokens with translations; programmatic substring check of every answerable item |
| W36 baselines | Individual (primary) | Majority-class baseline and the deterministic lexical-overlap rule (`answerable` iff at least one shared distinct token after NFC normalisation, case folding and Unicode tokenisation) |
| W37 span prediction | Individual | Character-offset to BIO conversion with the required edge-case checks; question-conditioned sequence labeller |
| W38 language modelling | Individual | Pure trigram model with add-k smoothing (k = 0.1), shared Korean vocabulary of 1,126 types and `<UNK>` mapping; perplexity 48.84 with a 38.29% unknown-token rate |
| W39 learned answerability | Individual | Logistic regression and BiLSTM classifiers under the shared protocol, with a ten-example, three-language error analysis across three failure categories |
| W40–41 integrated QA | Individual | Fine-tuning and validation-only threshold selection for the multilingual reader |
| **W43 extension: calibrated abstention** | Individual, **primary responsibility** | Falsifiable hypothesis that language-specific thresholds beat a pooled threshold; six decision rules (pooled, language-specific, temperature-scaled, recall-targeted, feature combiner, fixed cut-off); macro-F1 rose from 0.8576 to 0.8673, with EM 0.5532 → 0.5584 |
| W43–44 challenge set | Individual | Five matched pairs (ten records), validated with the starter script |
| Writing and code | Individual | Full report prose, figures/tables, and all submitted code |

*Generative AI:* $$
state your declaration here — e.g. "used for language editing and debugging only, with complete logs in `genai-logs/`", or "No generative AI was used in this project"
$$ — consistent with the standard declaration template and log requirements.

## If this was actually a group submission

Keep the table but change the first sentence to "The core pipeline (Weeks 36–41) was developed jointly by [members]. I took primary responsibility for the calibrated-abstention extension (W43) and wrote [sections]." Then mark shared rows as *Joint (with [name])* and keep only your extension and challenge pairs as yours. The instructions explicitly allow individual grades to differ when the statement shows materially different contributions, so being precise about the split matters more than the wording.[^1_1]

## Two wording cautions

Your report's W39 discussion is careful about causal claims, so mirror that restraint here — say you "analysed" and "compared", not that you "established" why the learned models beat the rule. Also mention the paired bootstrap interval [−0.0069, 0.0269] once in the statement, since it shows the extension's gain is not established on unseen data and demonstrates honest reporting to the examiner.

Was this an individual submission, or were you one of two or three members? I can rewrite it as a group statement and generate the LaTeX block for the ACL template.

<span style="display:none">[^1_2]</span>

<div align="center">⁂</div>

[^1_1]: instructions.pdf

[^1_2]: individual_report.pdf


---

# Can you give it to me in LaTeX, here is my report code:

```
\documentclass[11pt]{article}

\usepackage[a4paper,margin=2.4cm]{geometry}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{graphicx}

\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}

\title{Individual Report}

\author{Kieran O'Neill \\\texttt{zcd890}\\}

\begin{document}
\maketitle

\section{W36: Data inspection and initial baseline}

\subsection{Baseline 2: deterministic lexical-overlap rule classifier}

\subsubsection*{Implementation}

\noindent
Before comparison, both texts are normalised with Unicode NFC normalisation and case folding. It is then tokenised using a Unicode-aware regular expression that extracts word-character sequences while excluding punctuation. 

\noindent 
For each example, let $Q$ be the set of distinct question tokens and $C$ the set of distinct context tokens. The classifier computes the number of shared tokens,
\[
s = |Q \cap C|.
\]
It predicts answerable if $s \geq 1$, and unanswerable otherwise:
\[
    \hat{y} =
        \begin{cases}
            \text{answerable}, & \text{if } |Q \cap C| \geq 1, \\
            \text{unanswerable}, & \text{otherwise.}
        \end{cases}
\]
The threshold of one shared token is fixed before the labeled validation evaluation. Repeated tokens are counted only once because and punctuation-only tokens, including question marks, do not create an overlap.

\subsubsection*{Results}

\noindent
The deterministic rule obtains an overall accuracy of 0.1524 and macro-F1 of 0.1408 (Table~\ref{table:week_36_baseline_results}). It has unanswerable recall of 0.9451 but answerable recall of only 0.0212. The same pattern occurs for all three languages, although Telugu has the highest macro-F1 (0.2265).

\begin{table}[!h]
    \centering
    \small
    \begin{tabular}{lrrrrrrrrr}
        \toprule
        & & & &
        \multicolumn{3}{c}{Answerable}
        & \multicolumn{3}{c}{Unanswerable} \\
        \cmidrule(lr){5-7}\cmidrule(lr){8-10}
        Group & $N$ & Accuracy & Macro-F1
        & P & R & F1
        & P & R & F1 \\
        \midrule
            Overall & 1155 & 0.1524 & 0.1408 & 0.7000 & 0.0212 & 0.0411 & 0.1378 & 0.9451 & 0.2405 \\
            ar & 415 & 0.1398 & 0.1291 & 1.0000 & 0.0165 & 0.0325 & 0.1271 & 1.0 & 0.2256 \\
            ko & 356 & 0.0562 & 0.0547 & 0.6000 & 0.0089 & 0.0175 & 0.0484 & 0.8947 & 0.0919 \\
            te & 384 & 0.2552 & 0.2265 & 0.6316 & 0.0412 & 0.0774 & 0.2356 & 0.9247 & 0.3755 \\
        \bottomrule
    \end{tabular}
    \caption{Validation results for the deterministic lexical-overlap answerability rule. P, R, and F1 denote precision, recall, and F1, respectively. Macro-F1 is the unweighted mean of the answerable and unanswerable F1 scores.}
    \label{table:week_36_baseline_results}
\end{table}

\subsubsection*{Analysis}

\noindent
As we can see, this rule has high recall for unanswerable examples primarily because it lacks lexical evidence for most question/context pairs. Questions are in Arabic, Korean, or Telugu while contexts are English, so direct word overlap is rarely captured in translation equivalent terms. Therefore, this rule incorrectly rejects most answerable examples. 

\noindent
It can only exploit surface forms shared across languages, such as Latin-script names and some numbers. It also produces false positives when an incidental shared surface token occurs although the context does not answer the question. This simple baseline clearly demonstrates why a cross-lingual answerability model needs a representation that can relate questions to English contexts.


\section{W38: Pure Trigram Language Model}

\subsubsection*{Implementation}

\noindent
The configuration is a pure trigram language model with add-$k$ smoothing, where $k=0.1$:
\[
P \left(
        w_{i} \mid w_{i-2}, w_{i-1}
    \right) = \frac{
        C \left(
            w_{i-2}, w_{i-1}, w_{i}
        \right) + k
    }{
        C \left(
            w_{i-2}, w_{i-1}
        \right) + k \vert V \vert
    }.
\]
The history count in the denominator counts occurrences of the two-token history with an outgoing prediction target.

\noindent
Rare training tokens and out-of-vocabulary validation tokens map to \texttt{<UNK>}. The prediction vocabulary contains 1,126 types, including \texttt{<UNK>} and \texttt{<EOS>} (\texttt{<BOS>} is intentionally left out). Each question receives two \texttt{<BOS>} history markers and one \texttt{<EOS>} target.  All question tokens and the final \texttt{<EOS>} are scored.

\noindent
The overall perplexity sums the negative log-probabilities across all targets rather than averaging question perplexities.

\subsubsection*{Results}

\begin{table}[!h]
    \centering
    \small
    \begin{tabular}{lr}
        \toprule
        Model & Perplexity \\
        \midrule    
        Pure trigram with add-$k$ smoothing & 48.8377 \\
        \bottomrule
    \end{tabular}
    \label{table:week_38_pure_trigram_perplexity}
\end{table}

\subsubsection*{Analysis}

\noindent
The 38.29\% unknown-token rate removes substantial lexical information, so low perplexity may partly reflect predictable unknown-token sequences rather than an accurate prediction of the original Korean words.

\section{W39: Learned Answerability}

\subsubsection*{Baseline 2 comparison}

\subsubsection*{Scope}

\noindent
The logistic-regression (LR) and BiLSTM classifiers were compared against the deterministic lexical-overlap baseline and the selected validation errors were analysed.

\subsubsection*{Results}

\begin{table}[!h]
    \centering
    \small
    \begin{tabular}{lrrrr}
        \toprule
        Model & Accuracy & Macro-F1 & Answerable F1 & Unanswerable F1 \\
        \midrule
        Majority baseline & 0.8580 & 0.4618 & 0.9236 & 0.0000 \\
        Lexical-overlap rule & 0.1524 & 0.1408 & 0.0411 & 0.2405 \\
        Logistic regression & 0.9126 & 0.8134 & 0.9494 & 0.6773 \\
        BiLSTM & 0.9143 & 0.8076 & 0.9509 & 0.6644 \\
        \bottomrule
    \end{tabular}
    \caption{Overall validation comparison.}
    \label{table:week_39_baseline_comparison}
\end{table}

\subsubsection*{Analysis}

\noindent
Both learned classifiers (LR and BiLSTM) substantially outperform the
lexical-overlap rule under macro-F1. The rule predicts answerable for only 30 examples: 21 true positives and nine false positives. It produces 970 false negatives, so its high unanswerable recall comes at the cost of rejecting almost all answerable examples.

\noindent
In contrast, LR and the BiLSTM achieve unanswerable F1 scores of 0.6773 and 0.6644 while maintaining answerable F1 scores above 0.94. This demonstrates the limitations of requiring identical surface tokens across non-English questions and English contexts, but does not establish that the learned models perform reliable cross-lingual semantic reasoning. LR has slightly higher macro-F1 despite slightly lower accuracy, illustrating why accuracy alone is insufficient.

\subsection*{Error analysis}

\subsubsection*{Scope}

\noindent
The ten validation examples selected: three Arabic, three Korean and four Telugu. The exmples were compared with the full English context and gold annotation, then each example was assigned one primary descriptive category.

\noindent
We define a false positive (FP) as an unanswerable example predicted answerable and a false negative (FN) as an answerable example predicted unanswerable.

\noindent
Six examples were misclassified by both models, two by LR and two by the BiLSTM. The ten distinct examples therefore contain sixteen model-specific errors, eight per classifier.

\subsubsection*{Results}

\begin{table}[!h]
    \centering
    \small
    \begin{tabular}{lrrr}
        \toprule
        Category & Cases & LR errors & BiLSTM errors \\
        \midrule
        Supported answer overlooked & 4 & 3 & 3 \\
        Requested specificity not established & 3 & 3 & 3 \\
        Ambiguous evidence or annotation & 3 & 2 & 2 \\
        \midrule
        Total & 10 & 8 & 8 \\
        \bottomrule
    \end{tabular}
    \caption{Primary categories within the selected sample. All errors are counted against the unchanged gold labels.}
    \label{table:week_39_error_categories}
\end{table}

\subsubsection*{Analysis}

Examples \textbf{48} and \textbf{221} (Telugu), \textbf{438} (Korean) and \textbf{752} (Arabic) contain support for the annotated answer but are rejected by at least one classifier.

\paragraph{Examples 48 and 221} These examples ask what the temperature is at which water becomes steam. Their identical context explicitly contains ``boiling point (100~$^\circ$C)''. However, LR rejects both, whereas BiLSTM accepts \textbf{48} but rejects \textbf{221}. This demonstrates sensitivity to question wording with the context held fixed, although it does not identify the mechanism. Thus, each classifier's gold spans differ in extent, despite their shared answerability label.

\paragraph{Example 438} 
For this example, the context states that Indonesian law ``does not recognise same-sex marriage'', supporting the translated question about legality. LR predicts answerable, but the BiLSTM produces an FN. Negation and legal paraphrasing are observable features of this case, not established causes of the error.

\paragraph{Example 752} 
For this example, both classifiers reject a passage explicitly describing Paul as ``an apostle'', qualified by ``though not one of the Twelve Apostles''. The prepared translations differ between ``apostles'' and ``disciples'', so the interpretation warrants caution.

Examples \textbf{337} (Korean), \textbf{831} (Arabic) and \textbf{1154} (Telugu)
are FPs under both models.

\paragraph{Example 337} 
This example asks the question as to whether laws differ between U.S. states. The context defines state law as the law of each separate state, but does not explicitly establish differences between those laws. This is a borderline distinction between plausible inference and explicit contextual support.

\paragraph{Example 831} 
This example asks the question why St. John's wort has its name. This leads to a possible mismatch because the English name (i.e. ``St. John's wort``) bares little resemblance to the Latin name and is translation-dependent.

\paragraph{Example 1154}
This example shows the clearest specificity mismatch. The question asks Guntur town's population in the 2011 census, but the context gives the population of Guntur district, not the town.

\noindent
Examples \textbf{630} (Korean), \textbf{1003} (Arabic) and \textbf{1072} (Telugu) require caution when interpreting errors against gold labels.


\paragraph{Example 630} 
This example asks for the average lifetime of stars. The context discusses red dwarfs and compares their lifespans with the Sun's expected 10 billion a year lifespan. The gold span, ``10-billion'', is not an explicit average across stars and omits its unit. LR is an FN under the gold label, while the BiLSTM is correct.

\paragraph{Example 1003} 
Thie example asks whether Messi held Spanish citizenship before joining Argentina's national team. Its gold span, ``Spanish descent'', does not explicitly imply citizenship. Both classifiers are FNs under the answerable gold label, but the context does not appear to establish the requested fact.

\paragraph{Example 1072} 
This example asks for Yanam's area. The context contains a detached ``30 sq km''
phrase and an incomplete area statement. LR follows the unanswerable gold label; the BiLSTM predicts answerable. The raw answer field is nonempty, but its start offset
is $-1$, so it is not a valid extractable gold span. The numerical phrase and corrupted wording make the context difficult to interpret confidently.

\section{W43: Calibrated Abstention Extension}

\subsubsection*{Scope and hypothesis}

\noindent
This extension investigates whether changing the abstention decision rule improves the integrated multilingual QA system. It compares a single pooled no-answer threshold with language-specific thresholds and additional post-hoc decision rules.

\noindent
The primary hypothesis was that language-specific thresholds would increase answerability macro-F1 relative to a pooled threshold, because a shared multilingual reader need not produce equally distributed no-answer scores between Arabic, Korean and Telugu. A secondary question asks whether temperature scaling improves probability calibration independently of the final answerability decisions.

\subsubsection*{Implementation}

\noindent
The extension uses the integrated model's saved predictions, containing 1,155 validation examples: 415 Arabic, 356 Korean and 384 Telugu.

\noindent
For each example, the no-answer margin is
\[
    s = s_{\mathrm{null}} - s_{\mathrm{best}}.
\]
Larger margins favour abstention. For a threshold $t$, the system returns
the best non-empty span when $s \leq t$ and an empty answer otherwise.

\noindent
The original pipeline's thresholded predictions are not reused. Instead,
span exact match and token F1 are recomputed from the saved best-span text
against the gold answer. Each candidate rule then determines whether that
span is returned. A correctly abstained unanswerable example receives an
exact-match and token-F1 score of one.

\noindent
Six decision rules were compared:
\begin{description}
    \item[A: Pooled threshold.]
    One threshold selected to maximise answerability macro-F1 across the
    pooled validation examples.

    \item[B: Language-specific thresholds.]
    Separate thresholds selected to maximise answerability macro-F1 within
    each language.

    \item[C: Temperature scaling.]
    A positive temperature $T$ fitted by minimising binary negative
    log-likelihood on validation labels. The transformed answerability score is
    \[
        p_T = \frac{1}{1+\exp(s/T)}.
    \]
    A pooled decision threshold is subsequently selected on this score.

    \item[D: Recall-targeted thresholds.]
    Language-specific thresholds selected to target approximately 90\%
    unanswerable recall, rather than maximise macro-F1.

    \item[E: Feature combiner.]
    A logistic-regression classifier using the margin, log context length,
    log question length, lexical overlap and predicted-span length.
    Length features use whitespace token counts. Five-fold cross-fitting
    produces out-of-fold answerability probabilities, on which a pooled
    threshold is selected.

    \item[F: Fixed probability threshold.]
    The temperature-scaled probability is compared with a fixed cutoff of
    0.5, without additional decision-threshold tuning.
\end{description}

\subsubsection*{Results}

\begin{table}[!h]
    \centering
    \small
    \begin{tabular}{lrrrrr}
        \toprule
        Rule & Macro-F1 & Answerable F1 & Unanswerable F1 & EM & Token F1 \\
        \midrule
        A: Pooled threshold       & 0.8576 & 0.9626 & 0.7525 & 0.5532 & 0.6465 \\
        B: Language-specific      & 0.8673 & 0.9639 & 0.7707 & 0.5584 & 0.6518 \\
        C: Temperature-scaled     & 0.8576 & 0.9626 & 0.7525 & 0.5532 & 0.6465 \\
        D: Recall-targeted        & 0.7077 & 0.8649 & 0.5506 & 0.5186 & 0.5910 \\
        E: Feature combiner       & 0.8622 & 0.9624 & 0.7619 & 0.5584 & 0.6508 \\
        F: Fixed probability cut  & 0.8421 & 0.9621 & 0.7220 & 0.5420 & 0.6376 \\
        \bottomrule
    \end{tabular}
    \caption{Overall validation results for the six abstention rules. EM and token F1 are integrated QA metrics including unanswerable examples.}
    \label{table:week_43_overall}
\end{table}

\noindent
Language-specific thresholds achieve the highest overall macro-F1, increasing it from 0.8576 to 0.8673. Unanswerable F1 increases from 0.7525 to 0.7707, while answerable F1 changes only slightly. Integrated QA exact match increases from 0.5532 to 0.5584 and token F1 from 0.6465 to 0.6518.

\begin{table}[!h]
    \centering
    \small
    \begin{tabular}{llrrrr}
        \toprule
        Language & Rule & $N$ & Macro-F1 & Answerable recall & Unanswerable recall \\
        \midrule
        ar & A & 415 & 0.9371 & 0.9752 & 0.9423 \\
        ar & B & 415 & 0.9428 & 0.9752 & 0.9615 \\
        ko & A & 356 & 0.8332 & 0.9822 & 0.6842 \\
        ko & B & 356 & 0.8691 & 0.9970 & 0.6316 \\
        te & A & 384 & 0.7939 & 0.9656 & 0.5591 \\
        te & B & 384 & 0.8006 & 0.9347 & 0.6344 \\
        \bottomrule
    \end{tabular}
    \caption{Pooled (A) and language-specific (B) threshold results. Answerable recall is the proportion of gold-answerable examples answered, not the proportion answered correctly.}
    \label{table:week_43_by_language}
\end{table}

\subsubsection*{Analysis}

\noindent
Language-specific thresholds achieve the highest macro-F1 (0.8673), but their effects differ by language. Arabic and Telugu benefit from more conservative decisions, whereas Korean improves by accepting more answerable examples despite lower unanswerable recall. The overall gain over the pooled threshold is modest (0.0098), and the paired bootstrap interval $[-0.0069,\,0.0269]$ includes zero, so improvement on unseen examples is not established.

\noindent
Temperature scaling reduces expected calibration error from 0.0623 to 0.0319 without changing the decisions after threshold retuning. This is expected as a monotonic transformation changes probability calibration but preserves the available decision boundaries.

\noindent
The recall-targeted rule increases unanswerable recall to 0.8963, but rejects more supported questions, reducing macro-F1 to 0.7077. The feature combiner also falls below the language-specific rule, suggesting that the simpler threshold adjustment provides the best observed trade-off in this experiment.

\subsubsection*{Limitations}

\noindent
All rules are tuned and evaluated on the same validation split, so their results may be optimistic, particularly for Korean with only 19 unanswerable examples. The bootstrap interval holds fitted decisions fixed and does not capture threshold-selection uncertainty or variation across reader-training seeds.

\noindent
The extension cannot correct an inaccurate best span, and its calibration measures answerability rather than answer correctness. The recall-targeted rule approaches, but does not strictly achieve, 90\% unanswerable recall.

\end{document}
```

Here's the statement as a LaTeX block that fits your existing preamble — it uses only `booktabs`, which you already load, so you can drop it in unchanged.

## Where to insert it

Paste it immediately before `\end{document}`, after the W43 Limitations subsection. The instructions require the contribution statement to be placed last, and it does not count toward the page limit.[^2_1]

```latex
% ---------------------------------------------------------------
% Contribution statement (placed last; not counted in page limit)
% ---------------------------------------------------------------
\section*{Contribution Statement}

\noindent
This project was carried out individually by Kieran O'Neill (zcd890).
I had no group members, so every core component (Weeks~36--41), the
paired challenge set and the individual extension were designed,
implemented, evaluated and written by me alone; nothing was completed
jointly.\footnote{Completed alone with the permission granted in the
project description, which allows individual submission although groups
of up to three are recommended.}
I therefore take full responsibility for all parts of the submitted
report, code and challenge set.

\begin{table}[!h]
    \centering
    \small
    \begin{tabular}{@{}p{0.30\textwidth}p{0.18\textwidth}p{0.44\textwidth}@{}}
        \toprule
        Component & Responsibility & Contribution \\
        \midrule
        W36 data inspection (Items 1--3)
            & Individual
            & Per-language and per-split example counts, answerability
              proportions and length statistics; top-5 question tokens with
              translations; programmatic substring check of every answerable
              item. \\
        W36 baselines
            & Individual (primary)
            & Majority-class baseline and the deterministic lexical-overlap
              rule, i.e.\ answerable iff at least one shared distinct token
              after NFC normalisation, case folding and Unicode-aware
              tokenisation. \\
        W37 span prediction
            & Individual
            & Character-offset to BIO conversion with the required edge-case
              checks (answer at offset 0, multi-token answer, adjacent
              punctuation, unanswerable example); question-conditioned
              sequence labeller. \\
        W38 language modelling
            & Individual
            & Pure trigram model with add-$k$ smoothing ($k=0.1$), shared
              Korean vocabulary of 1{,}126 types and \texttt{<UNK>} mapping;
              perplexity 48.84 at a 38.29\% unknown-token rate. \\
        W39 learned answerability
            & Individual
            & Logistic-regression and BiLSTM classifiers evaluated under the
              shared protocol, plus a ten-example, three-language error
              analysis organised into three failure categories. \\
        W40--41 integrated QA
            & Individual
            & Fine-tuning of the multilingual reader and validation-only
              no-answer threshold selection. \\
        W43 extension: calibrated abstention
            & Individual (\textbf{primary})
            & Stated falsifiable hypothesis that language-specific thresholds
              improve on a pooled threshold; implemented and compared six
              decision rules (pooled, language-specific, temperature-scaled,
              recall-targeted, feature combiner, fixed cut-off). Macro-F1 rose
              from 0.8576 to 0.8673 and EM from 0.5532 to 0.5584, with a paired
              bootstrap interval of $[-0.0069,\,0.0269]$. \\
        W43--44 challenge set
            & Individual
            & Five matched pairs (ten records), validated with the supplied
              \texttt{validate\_challenge.py}. \\
        Writing and code
            & Individual
            & All report prose, tables and the submitted code. \\
        \bottomrule
    \end{tabular}
    \caption{Division of work. Every component was completed individually,
    as the project was submitted by a single author.}
    \label{table:contribution_statement}
\end{table}

\noindent
\textbf{Generative AI.}
\textit{[State your declaration here, matching the standard template:
for example ``Generative AI was used for language editing and debugging
only; complete prompt and output logs are provided in the
\texttt{genai-logs} directory'', or ``No generative AI was used in this
project, so no logs are provided.'']}
```


## Two adjustments worth making

Delete the footnote if your report never mentions submitting solo — it draws attention to something the examiner may not have flagged, and the table caption already says "completed individually". Keep the \\textbf{primary} marker on the W43 row, since the instructions specifically ask who holds primary responsibility for each extension.[^2_1]

Replace the bracketed generative-AI sentence with the exact wording of the official declaration template rather than my paraphrase; the instructions require that standard form and complete logs in a `genai-logs` directory if AI was used at all. Any undeclared AI-assisted prose or code in your report is treated as a code-of-conduct issue, so err toward over-disclosure.[^2_1]

If the report is in fact one chapter of a group submission, tell me the other members and I'll convert the middle column to joint/individual and rewrite the opening paragraph.

<div align="center">⁂</div>

[^2_1]: instructions.pdf

