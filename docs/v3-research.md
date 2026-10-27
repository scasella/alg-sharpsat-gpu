# V3 source research: the mechanism missing from V2

Research date: 27 October 2026. This is an implementation comparison, not an independent verification of the claimed asymptotic theorem. No V3 code design was committed while obtaining the sources.

## Primary sources actually read

1. [Ryan O'Donnell, *Some exposition of “The Exponential Time Hypothesis Is False”*](https://www.cs.cmu.edu/~odonnell/some_eth_false_exposition.pdf), all 12 pages. The title page is dated **26 October 2026**; the author's public post was dated 27 October 2026, 02:04 UTC. The footnote attributes production to interaction with LLMs. This is the requested primary exposition, not a substitute article. Its independent-set specialization has different parameter names from the manuscript.
2. [OpenAI, *The Exponential Time Hypothesis is False*, revised 25 October 2026](https://github.com/openai/math/blob/main/preprints/The-Exponential-Time-Hypothesis-is-False-October-25-2026/paper.pdf), relevant complete Sections 2.1–2.6, 3.3–3.5, and their equations, pp. 7–13 and 22–26. This comparison covers the sections listed here; it does not independently verify the entire manuscript.

The exposition URL was recovered from the full link behind the author's quoted post, then downloaded directly from CMU. Its text and the complete rendered pages 7–10 were inspected. An indexed social mirror was used only to discover the URL, not as mathematical evidence.

## The exact difference

V2 assigns each boundary link one degree-one root. The sets of links suppressed at the evaluation nodes partition the links. Its remaining coupling is represented by measurement values plus explicitly retained master bits. This is an exact construction, but it has no mechanism making every node cheap. If there are B links and q_s is the maximum retained jet degree, then

\[
\sum_s|K_s|=B,\qquad \sum_s(q_s+1)=B+1
\quad\Longrightarrow\quad \exists s:q_s\ge |K_s|.
\]

At that node the suppressed-link expansion includes every subset of K_s. More nodes or a different distribution of orders cannot remove this obstruction within that model. This conclusion follows from V2's definitions, not from a lower bound on the manuscript's algorithm.

The primary exposition uses a different object: **dense occurrence-specific linear replacements with a shared exact low-rank correction**, accompanied by different variable partitions and factor assignments at different nodes. It combines two properties of the same matrix, not just zeros assigned to existing links. [Exposition, Lemma 3.1 and Eqs. (3.3)–(3.8), pp. 3–4; Lemma 4.1, pp. 4–5.](https://www.cs.cmu.edu/~odonnell/some_eth_false_exposition.pdf#page=3)

To avoid notation collisions below, v is the number of graph vertices; Q is the exposition's number n of register pairs and evaluation nodes; d is the number of groups; b bounds replacement degree; N=d^(2Q) is auxiliary matrix dimension.

1. Let S_i swap one register pair, W=Σ_i(I−S_i)/2, U=∏_i S_i, and M=U f(W). The degree-b rational polynomial f agrees with parity on typical binomial-weight layers. Consequently rank(I−M)≤ρN, while every nonzero M entry differs from a complete swap at at most b locations. The exceptional spectral directions remain as explicit {0,±1} measurement rows C. They are **retained exactly**, not discarded as an approximation. [Exposition, Lemmas 2.1 and 3.1, Eqs. (2.1), (3.4)–(3.9), pp. 2–4.](https://www.cs.cmu.edu/~odonnell/some_eth_false_exposition.pdf#page=2)
2. For a row ω and column η, let B_(ω,η) be the bad locations and φ_(ω,η)(T)=∏_(t∈B_(ω,η))(1−T/t). The replacement is

   \[
   K_\omega(T,z)=((I-M)z)_\omega+
       \sum_\eta M_{\omega,\eta}\phi_{\omega,\eta}(T)z_\eta.
   \]

   It restores z_ω at zero and has degree at most b. At each nonzero node, **all surviving noncorrection coordinates lie in the factor's assigned group**; C z determines the correction. Many matrix entries can vanish at a node even though each coefficient polynomial has small degree. V2's partition of one root per existing link does not express this matrix structure. [Exposition, Eqs. (4.1)–(4.2), Lemma 4.1, pp. 4–5.](https://www.cs.cmu.edu/~odonnell/some_eth_false_exposition.pdf#page=4)
3. Occurrence positions contain repeated values of original variables, not new independent Boolean variables. Each global measurement must be split into its restrictions to the node's groups. Fixing only a total measurement can still couple groups. [Exposition, Sections 5.2–5.3, Eqs. (5.6)–(5.9), pp. 6–7.](https://www.cs.cmu.edu/~odonnell/some_eth_false_exposition.pdf#page=6)

## Section 6: the separated evaluator and its state count

For fixed node t and group measurements y, define X_g(y_g)={u:C_(t,g)u=y_g}. Representatives in the other groups let one evaluate a factor assigned to group g on a local u. They need satisfy the measurements, **not** any additional fixed-bit restrictions. The exact evaluator is

\[
H(t)=\sum_y\prod_g\left(\sum_{u\in X_g(y_g)}F_{t,g}(u;y)\right).
\]

There are at most (2N+1)^s measurement tuples, s≤6ρdv, and at most 2d·2^(2v/d) local assignment visits per tuple, including representative selection. The comparison is a **sum of local enumeration sizes**, not the Cartesian product of every group's local assignments. Polynomial-valued fixed input shifts and local restrictions x_i=1 preserve this evaluator. [Exposition, Section 6, Eqs. (6.1)–(6.4), pp. 7–8.](https://www.cs.cmu.edu/~odonnell/some_eth_false_exposition.pdf#page=7)

For the manuscript's general design, let c be the total number of measurement rows, D their maximum coefficient ℓ1 norm, and S_g the groups. Lemma 3.2 gives at most

\[
2(2D+1)^c\sum_g2^{|S_g|}
\]

local Boolean visits per restricted sum. Eq. (3.4) specifies its group accumulator; the final update is Z←Z+∏_gΣ_g(y). This is an explicit nested summation routine, not an unstated recursive SAT recurrence. [Manuscript, Section 3.3, Lemma 3.2 and Eq. (3.4), pp. 22–23.](https://github.com/openai/math/blob/main/preprints/The-Exponential-Time-Hypothesis-is-False-October-25-2026/paper.pdf#page=22)

## Taylor recovery uses low-weight original-variable substitutions

The exposition's Section 7, rather than Section 6, supplies derivative recovery. Its degree bound is deg H≤3vb, with q=⌊3vb/Q⌋. Introduce a second **original-variable** vector X only in the correction L(t+z,X)−L(t,X). Each X occurrence carries at least one z, so the coefficient of z^j has X-degree at most j. Boolean Möbius inversion then gives

\[
H(t+z)\equiv\sum_{|A|\le q}\sum_{B\subseteq A}
(-1)^{|A|-|B|}
\sum_{x:x_i=1\;(i\in A)}F_{t,x}(1_B,z)
\pmod {z^{q+1}}.
\]

Every inner sum retains Section 6's separation: X=1_B makes the correction a fixed polynomial shift. Its number of (A,B) pairs is

\[
C_q=\sum_{j=0}^{\min(q,v)}2^j\binom vj,
\]

not a count of differentiated boundary-link subsets. In particular C_q=3^v when q≥v; a small finite prototype may be slower than exhaustive counting. [Exposition, Eqs. (7.1)–(7.11), pp. 8–10; manuscript, Lemma 3.3, Eqs. (3.10), (3.13)–(3.14), pp. 24–25.](https://www.cs.cmu.edu/~odonnell/some_eth_false_exposition.pdf#page=8)

The relevant finite-size work bound is therefore polynomial overhead times

\[
Q\,C_q\,(2D+1)^c\sum_g2^{|S_g|}.
\]

Within each local factor product, truncated coefficients obey the ordinary recurrence a'_[j]=Σ_(ℓ=0)^j a_[ℓ]f_[j−ℓ]. It uses O(q²) operations per multiplication and q+1 stored coefficients. These operations do not enumerate multivariate monomials. Proposition 3.4 includes the loop over Q nodes and all (A,B) pairs. The asymptotic gain depends on simultaneously making b/Q, the measurement cost, and group size small; Hermite interpolation alone supplies none of these. [Manuscript, Proposition 3.4 and Section 3.5, pp. 25–26; exposition, Eq. (8.1), p. 11.](https://github.com/openai/math/blob/main/preprints/The-Exponential-Time-Hypothesis-is-False-October-25-2026/paper.pdf#page=25)

## A faithful finite implementation to investigate

The following is an engineering inference from the identities, not an assertion that the source's asymptotic parameter regime is practical.

* Represent a design explicitly: occurrence replacement coefficient vectors, a partition and factor assignment per node, and group-restricted integer measurement matrices. Verify restoration at T=0 and row-space/locality identities exactly.
* Construct a tiny actual swap-polynomial matrix first. For example d=2, Q=2, b=1, f(w)=1−2w gives N=16 and rank(I−M)=1: only the doubly antisymmetric eigenspace is exceptional. One C row is (|01⟩−|10⟩)⊗². This toy does not meet Lemma 2.1's asymptotic b≥32 assumption; its finite identities can instead be verified directly. It tests dense replacements plus exact exceptional measurements, which V2 cannot represent.
* Implement the separated evaluator with fixed shifts and original-variable restrictions, then the (A,B) Möbius sum and existing exact Hasse/Hermite recovery. Retain all denominator conditions when using modular arithmetic: chosen primes must make rational coefficients meaningful and nodes distinct. A prime greater than the polynomial degree is unnecessary for Hasse interpolation itself.
* Validate locality by varying other-group assignments with equal measurements; compare small H(t), individual jets, and final integer counts against independent scalar oracles. Exhaustive oracles belong in validation, not as a hidden runtime fallback.
* Report Q, replacement degree, true measurement ranges and feasible tuples, each group size, q, C_q, local visits, and rational denominator exclusions. A smaller degree or fewer link subsets is not evidence of the source's complexity mechanism by itself.

Adapting the general manuscript's factor formulation to CNF is closer to the existing interface than first converting to cubic-graph independent sets. The exposition remains useful because its matrix and separated-evaluator equations make the missing structure explicit. Full sparsification and the sharper counting reductions are separate obligations; reproducing this bounded kernel would not establish the manuscript's complete theorem.

## Implemented finite kernel

V3 now includes both an independent multilinear reference and a default
occurrence-coherent clause evaluator. In `OccurrenceSwapCounter`, each original
variable has one first-register word across all auxiliary blocks, and each
clause has one second-register word. Conflicts at an auxiliary coordinate are
placed in additional blocks. Occupied coordinates read their original Boolean
variables; unused coordinates are fixed zero. Measurements are expanded through
this occurrence map, then split by each node's original-variable groups.
The default evaluator retains clause products and implements the restricted
local sums and original-variable Möbius substitutions directly.

This finite cyclic labeling is an engineering construction satisfying the
checked restoration and locality equations. It does not implement the
asymptotic incidence design, its balancing/rank guarantees, full sparsification,
or polynomial-space enumeration. The older multilinear counter remains an
independent comparison and explicitly pays for its monomial expansion.
The benchmark distinguishes both representations and uses occurrence-specific
cost bounds for the default path. A subsequent review compared previously
retrieved primary-source copies against these identities. That review could not
re-fetch the CMU URL; the original retrieval and review scope are described above.
