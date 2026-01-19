import streamlit as st
import pandas as pd
from genesyn_core import generate_gene, score_gene
from genesyn_predict import predict_score

st.set_page_config(
    page_title="GeneSyn – Synthetic Gene Designer",
    layout="wide"
)

st.title("🧬 GeneSyn – AI-Assisted Synthetic Gene Designer")
st.markdown(
    """
    **GeneSyn** designs and ranks optimized synthetic genes for *E. coli*  
    using **biological rules + AI (BiLSTM)**.
    """
)

# -----------------------------
# INPUT SECTION
# -----------------------------
st.sidebar.header("🔹 Input")

protein_seq = st.sidebar.text_area(
    "Paste Protein Sequence (FASTA without >)",
    height=200,
    placeholder="Example: MKVLYNL..."
)

num_variants = st.sidebar.slider(
    "Number of Gene Variants",
    min_value=5,
    max_value=50,
    value=20
)

optimize_btn = st.sidebar.button("🚀 Optimize Gene")

# -----------------------------
# MAIN PROCESS
# -----------------------------
if optimize_btn:

    if len(protein_seq.strip()) == 0:
        st.error("❌ Please enter a protein sequence.")
    else:
        st.info("🔄 Generating and optimizing gene variants...")

        results = []

        for i in range(num_variants):
            # Generate synonymous gene
            dna_seq = generate_gene(protein_seq)

            # Rule-based scoring
            cai, gc, penalty, rule_score = score_gene(dna_seq)

            # AI-based prediction
            ai_score = predict_score(dna_seq)

            results.append([
                dna_seq,
                round(cai, 3),
                round(gc, 2),
                penalty,
                round(rule_score, 3),
                round(ai_score, 3)
            ])

        # Create DataFrame
        df = pd.DataFrame(
            results,
            columns=[
                "DNA Sequence",
                "CAI",
                "GC %",
                "Penalties",
                "Rule-Based Score",
                "AI Score"
            ]
        )

        # Rank by AI Score primarily
        df = df.sort_values(
            by=["AI Score", "Rule-Based Score"],
            ascending=False
        ).reset_index(drop=True)

        df["Rank"] = df.index + 1

        # -----------------------------
        # OUTPUT SECTION
        # -----------------------------
        st.success("✅ Optimization Completed")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📊 Ranked Gene Candidates")
            st.dataframe(df, use_container_width=True)

        with col2:
            best_gene = df.iloc[0]["DNA Sequence"]

            st.subheader("🏆 Best Gene (AI-Recommended)")
            st.text_area(
                "FASTA Output",
                f">GeneSyn_Best_Ecoli\n{best_gene}",
                height=200
            )

            st.download_button(
                label="⬇️ Download FASTA",
                data=f">GeneSyn_Best_Ecoli\n{best_gene}",
                file_name="genesyn_best_gene.fasta",
                mime="text/plain"
            )

        # -----------------------------
        # EXPLANATION SECTION (VIVA BONUS)
        # -----------------------------
        with st.expander("📘 How GeneSyn Optimizes Genes (For Viva)"):
            st.markdown(
                """
                **Step 1 – Variant Generation**  
                Multiple synonymous DNA sequences are generated from the input protein
                using the genetic code.

                **Step 2 – Rule-Based Optimization**  
                Each gene is evaluated using:
                - Codon Adaptation Index (CAI)
                - GC Content
                - Sequence penalties (repeats, inefficiencies)

                **Step 3 – AI-Based Ranking**  
                A **BiLSTM neural network**, trained on synthetic codon variants,
                predicts an independent optimization score from the DNA sequence.

                **Final Selection**  
                Genes are ranked using both rule-based and AI scores.
                The top-ranked gene is recommended for *E. coli* expression.
                """
            )

else:
    st.info("👈 Enter a protein sequence and click **Optimize Gene** to start.")
