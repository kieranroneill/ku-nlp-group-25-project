<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# I have have included the instructions for an assignment. For my Week 43 Individual Extension contribution, I have written the following document:

```
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
```

However, it is too long and needs to be trimmed down as we only have six pages (in ACL format).

Can you make this more concise, without losing information and in exactly the same LaTeX format?

Here is a more **concise** version that retains both tables, all six decision rules, the hypothesis, implementation details, findings and limitations. It keeps your existing LaTeX structure while reducing repetition to help fit the six-page report limit.[^1_1]

```latex
\section{W43: Calibrated Abstention Extension}

\subsubsection*{Scope and hypothesis}

\noindent
We test whether language-specific abstention thresholds improve
answerability macro-F1 over a pooled threshold, since the multilingual
reader's no-answer score distributions may differ across Arabic, Korean
and Telugu. We also compare alternative decision rules and ask whether
temperature scaling improves probability calibration independently of
answerability decisions.

\subsubsection*{Implementation}

\noindent
We use saved integrated-model predictions for 1,155 validation examples
(415 Arabic, 356 Korean, 384 Telugu). The no-answer margin is
\[
    s = s_{\mathrm{null}} - s_{\mathrm{best}}.
\]
The system returns the best non-empty span when $s \leq t$ and abstains
otherwise. We discard the original thresholded predictions and recompute
span EM and token F1 from saved best-span text against gold answers,
then apply each rule. Correct abstentions receive EM and token F1 of one.

\noindent
We compare six rules:
\begin{description}
    \item[A: Pooled threshold.]
    One threshold maximising pooled validation answerability macro-F1.

    \item[B: Language-specific thresholds.]
    Separate thresholds maximising macro-F1 within each language.

    \item[C: Temperature scaling.]
    A positive temperature $T$ minimises validation binary negative
    log-likelihood, producing answerability probabilities
    \[
        p_T = \frac{1}{1+\exp(s/T)}.
    \]
    A pooled decision threshold is then tuned.

    \item[D: Recall-targeted thresholds.]
    Language-specific thresholds targeting approximately 90\%
    unanswerable recall instead of macro-F1.

    \item[E: Feature combiner.]
    Logistic regression using margin, log context and question lengths,
    lexical overlap and predicted-span length, with whitespace token
    counts for lengths. Five-fold cross-fitting produces out-of-fold
    answerability probabilities for pooled threshold selection.

    \item[F: Fixed probability threshold.]
    Temperature-scaled probabilities with a fixed 0.5 cutoff and no
    decision-threshold tuning.
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
    \caption{Overall validation results. EM and token F1 include
    unanswerable examples; $N=1{,}155$.}
    \label{table:week_43_overall}
\end{table}

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
    \caption{Pooled (A) versus language-specific (B) thresholds.
    Answerable recall measures whether gold-answerable examples are
    answered, not whether their answers are correct.}
    \label{table:week_43_by_language}
\end{table}

\subsubsection*{Analysis}

\noindent
Language-specific thresholds perform best: macro-F1 rises from 0.8576
to 0.8673, chiefly through higher unanswerable F1, while EM and token F1
also improve (Table~\ref{table:week_43_overall}). Arabic and Telugu
benefit from more conservative decisions; Korean instead accepts more
answerable examples at the cost of unanswerable recall
(Table~\ref{table:week_43_by_language}). However, the macro-F1 gain
(0.0098) has a paired bootstrap interval of $[-0.0069,\,0.0269]$,
so improvement on unseen examples is not established.

\noindent
Temperature scaling reduces expected calibration error from 0.0623
to 0.0319 but leaves decisions unchanged after threshold retuning:
the monotonic transformation preserves available decision boundaries.
Recall targeting raises unanswerable recall to 0.8963, just below 90\%,
but rejects more supported questions, lowering macro-F1 to 0.7077.
The feature combiner also trails language-specific thresholds, which
provide the best observed trade-off.

\subsubsection*{Limitations}

\noindent
Tuning and evaluation share the validation split, potentially inflating
results, especially for Korean's 19 unanswerable examples. The bootstrap
holds fitted decisions fixed, excluding threshold-selection uncertainty
and reader-training seed variation. Abstention rules cannot repair
incorrect best spans, and calibration concerns answerability, not
answer correctness.
```

<div align="center">⁂</div>

[^1_1]: instructions.pdf


---

# Can you make it even more concise, it seems still too long. And use thrid-person perspective.

This version uses **third-person** throughout and compresses the prose further while retaining the numerical results and methodological caveats needed for the report.[^2_1]

```latex
\section{W43: Calibrated Abstention Extension}

\subsubsection*{Scope and hypothesis}

\noindent
This extension tests whether language-specific thresholds improve
answerability macro-F1 over a pooled threshold, given potentially
different no-answer score distributions across languages. It also
compares alternative abstention rules and tests whether temperature
scaling improves calibration independently of decisions.

\subsubsection*{Implementation}

\noindent
Saved integrated-model predictions cover 1,155 validation examples
(415 Arabic, 356 Korean, 384 Telugu). With no-answer margin
\[
    s = s_{\mathrm{null}} - s_{\mathrm{best}},
\]
the system returns the best span when $s \leq t$ and abstains otherwise.
EM and token F1 are recomputed against gold answers from saved
best-span text, replacing the original thresholded outputs; correct
abstentions score one.

\noindent
Six rules are compared:
\begin{description}
    \item[A: Pooled threshold.]
    Maximises pooled validation answerability macro-F1.
    \item[B: Language-specific thresholds.]
    Maximise macro-F1 separately per language.
    \item[C: Temperature scaling.]
    Fits $T>0$ by validation binary negative log-likelihood:
    \[
        p_T = \frac{1}{1+\exp(s/T)}.
    \]
    A pooled probability threshold is then tuned.
    \item[D: Recall-targeted thresholds.]
    Target approximately 90\% unanswerable recall per language.
    \item[E: Feature combiner.]
    Logistic regression uses margin, log question/context lengths,
    lexical overlap and span length; lengths count whitespace tokens.
    Five-fold out-of-fold probabilities support pooled threshold tuning.
    \item[F: Fixed probability threshold.]
    Applies a fixed 0.5 cutoff to $p_T$ without threshold tuning.
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
    \caption{Validation results ($N=1{,}155$); QA metrics include
    unanswerable examples.}
    \label{table:week_43_overall}
\end{table}

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
    \caption{Pooled (A) versus language-specific (B) thresholds.
    Answerable recall measures answering, not answer correctness.}
    \label{table:week_43_by_language}
\end{table}

\subsubsection*{Analysis}

\noindent
Rule B performs best, mainly improving unanswerable F1 with little
change in answerable F1; EM and token F1 also increase.
Arabic and Telugu favour more conservative decisions, whereas Korean
gains by answering more despite lower unanswerable recall.
The macro-F1 gain (0.0098) has a paired bootstrap interval of
$[-0.0069,\,0.0269]$, leaving generalisation unestablished.

\noindent
Temperature scaling lowers expected calibration error from 0.0623
to 0.0319 without changing retuned decisions, since monotonic
transformations preserve decision boundaries. Rule D reaches 0.8963
unanswerable recall, short of 90\%, but rejects supported questions,
reducing macro-F1 to 0.7077. Rule E also trails B.

\subsubsection*{Limitations}

\noindent
Shared tuning/evaluation data risk optimism, especially with only
19 Korean unanswerable examples. The bootstrap fixes fitted decisions,
excluding threshold-selection uncertainty and reader-seed variation.
Abstention cannot repair incorrect spans; calibration measures
answerability, not answer correctness.
```

<div align="center">⁂</div>

[^2_1]: instructions.pdf

