import random
import numpy as np

# Simplified E.coli codon table
CODON_TABLE = {
    'A': ['GCT','GCC','GCA','GCG'],
    'C': ['TGT','TGC'],
    'D': ['GAT','GAC'],
    'E': ['GAA','GAG'],
    'F': ['TTT','TTC'],
    'G': ['GGT','GGC','GGA','GGG'],
    'H': ['CAT','CAC'],
    'I': ['ATT','ATC','ATA'],
    'K': ['AAA','AAG'],
    'L': ['TTA','TTG','CTT','CTC','CTA','CTG'],
    'M': ['ATG'],
    'N': ['AAT','AAC'],
    'P': ['CCT','CCC','CCA','CCG'],
    'Q': ['CAA','CAG'],
    'R': ['CGT','CGC','CGA','CGG','AGA','AGG'],
    'S': ['TCT','TCC','TCA','TCG','AGT','AGC'],
    'T': ['ACT','ACC','ACA','ACG'],
    'V': ['GTT','GTC','GTA','GTG'],
    'W': ['TGG'],
    'Y': ['TAT','TAC']
}

def generate_gene(protein):
    dna = ""
    for aa in protein:
        dna += random.choice(CODON_TABLE.get(aa, ['ATG']))
    return dna

def gc_content(seq):
    return (seq.count('G') + seq.count('C')) / len(seq) * 100

def penalty(seq):
    return seq.count("AAAA") + seq.count("TTTT")

def score_gene(seq):
    gc = gc_content(seq)
    pen = penalty(seq)
    cai = random.uniform(0.6, 1.0)  # simplified CAI
    score = (1.0 * cai) - (0.1 * abs(gc - 50)) - (0.5 * pen)
    return cai, gc, pen, score
