"""
Constants for Xenium pancreas dataset.
"""

XENIUM_CELL_TYPES = [
    "Acinar",
    "B Cells",
    "CFTR- Tumor Cells",
    "CXCL9/10 Cells",
    "Ductal",
    "Endocrine 1",
    "Endocrine 2",
    "Endothelial",
    "Fibroblasts",
    "Lymphatic Endothelial Cells",
    "Macrophages",
    "Mast Cells",
    "Metaplastic Cells",
    "Smooth Muscle Cells",
    "T Cells",
    "Tumor Cells",
]

XENIUM_GENES = [
    "ABCC11", "ACE2", "ACKR1", "ACTA2", "ACTG2", "ACTN1", "ADAM28", "ADAMTS1", "ADGRE1", "ADGRL4",
    "ADH1C", "ADH4", "ADIPOQ", "AGER", "AGR3", "AHSP", "AIF1", "AKR1C3", "ALAS2", "ALDH1A3", "ALOX5AP",
    "AMY2A", "ANGPT2", "ANPEP", "ANXA3", "APCDD1", "APOA5", "APOBEC3A", "APOLD1", "AQP2", "AQP3",
    "AQP8", "AQP9", "AR", "ARFGEF3", "ASAH1", "ASCL1", "ASCL3", "ASPN", "ATP5F1B", "ATP5MC2", "ATP5MD",
    "BAMBI", "BANK1", "BASP1", "BBOX1", "BCL2L11", "BIRC3", "BMX", "BTF3", "BTNL9", "C15orf48", "C1R",
    "C1orf162", "C1orf194", "C20orf85", "C5orf46", "C6orf118", "C7", "CA1", "CA4", "CALU", "CAP1",
    "CAPN8", "CAST", "CAV1", "CAVIN1", "CAVIN2", "CCDC39", "CCDC78", "CCL19", "CCL27", "CCL5", "CCNB2",
    "CCR2", "CCR7", "CD14", "CD163", "CD19", "CD1A", "CD1C", "CD1E", "CD2", "CD247", "CD27", "CD274",
    "CD28", "CD300E", "CD34", "CD3D", "CD3E", "CD4", "CD5L", "CD68", "CD69", "CD70", "CD79A", "CD83",
    "CD86", "CD8A", "CD93", "CDH16", "CDK1", "CENPF", "CFAP53", "CFB", "CFHR1", "CFHR3", "CFTR", "CHGA",
    "CIB1", "CLCA1", "CLCA2", "CLEC10A", "CLEC14A", "CLEC4E", "CLECL1", "CLIC6", "CNN1", "CNN3", "COCH",
    "COL17A1", "COL5A2", "COMMD6", "CPA3", "CRHBP", "CRISPLD2", "CSF2RA", "CSF3", "CTHRC1", "CTLA4",
    "CTSG", "CTSK", "CXCL10", "CXCL2", "CXCL6", "CXCL9", "CXCR4", "CYP1A1", "CYP2A7", "CYP2B6",
    "CYP2F1", "CYP3A4", "CYP4B1", "CYTIP", "DERL3", "DES", "DIRAS3", "DKK3", "DLK1", "DMBT1", "DMKN",
    "DNAAF1", "DNASE1L3", "DPEP1", "DPT", "DSP", "DST", "DUSP1", "DUSP2", "ECSCR", "EDN1", "EDNRB",
    "EEF1B2", "EEF1D", "EGFL7", "EGFR", "EHF", "EIF2S2", "EIF4A1", "EIF4EBP1", "ELF5", "EMP3", "EPAS1",
    "EPCAM", "ERBB2", "ERG", "ESM1", "ESR1", "F3", "FAS", "FBLN1", "FBN1", "FCER1A", "FCGR1A", "FCGR3A",
    "FCN1", "FCN2", "FGFBP1", "FGFBP2", "FGL2", "FHL1", "FHL2", "FKBP11", "FLNA", "FOXA1", "FOXI1",
    "FOXJ1", "FOXP3", "FSTL1", "FSTL3", "FXYD2", "FXYD5", "GADD45A", "GATA2", "GATM", "GCG", "GDF15",
    "GEM", "GHRL", "GKN2", "GLIPR1", "GLYATL1", "GNAS", "GNG11", "GNLY", "GPC1", "GPC3", "GPR183",
    "GPRC5A", "GPX2", "GSTA1", "GYPA", "GYPB", "GZMA", "GZMB", "GZMK", "HADHB", "HAMP", "HAVCR2",
    "HEMGN", "HEPACAM2", "HES4", "HIGD1B", "HINT1", "HLA-DQB2", "HMGB2", "HMGCS2", "HNRNPA2B1", "HPGDS",
    "HPX", "HRC", "HSP90B1", "HSPA8", "IGF1", "IGSF6", "IL1B", "IL1R2", "IL1RL1", "IL2RA", "IL3RA",
    "IL7R", "INMT", "INS", "IRF8", "KCNK3", "KCNMA1", "KIT", "KLF6", "KLK11", "KLRB1", "KLRC1", "KLRD1",
    "KNG1", "KRT20", "KRT7", "KRTAP2-3", "LAG3", "LAMB3", "LAMP3", "LAPTM5", "LGI4", "LGR5", "LIF",
    "LILRA4", "LILRA5", "LILRB2", "LILRB4", "LMOD2", "LPL", "LSP1", "LTBP2", "LY6D", "LY86", "LYVE1",
    "MALL", "MAMDC2", "MARCO", "MAT1A", "MCEMP1", "MCF2L", "MDM2", "MEDAG", "MEF2C", "MEST", "MET",
    "MFAP5", "MKI67", "MLANA", "MLPH", "MMRN1", "MMRN2", "MNDA", "MORF4L1", "MPEG1", "MRC1", "MS4A1",
    "MS4A2", "MS4A4A", "MS4A6A", "MTRNR2L11", "MYBPC1", "MYC", "MYDGF", "MYH11", "MYH9", "MYLK",
    "MYOM1", "MZB1", "NAT8", "NDUFC2", "NKG7", "NNMT", "NOP53", "NPC2", "NPDC1", "NTN4", "NUPR1", "OGN",
    "OPRPN", "OST4", "OSTC", "PABPC1", "PCNA", "PCOLCE", "PCP4", "PCSK2", "PDCD1", "PDGFRA", "PDGFRB",
    "PDPN", "PEBP4", "PECAM1", "PFDN5", "PGC", "PGR", "PIP", "PLA2G7", "PLAC9", "PLCG2", "PLD4",
    "PLIN4", "PMP22", "PPA1", "PPARG", "PPP1R12B", "PPP1R1A", "PPP1R1B", "PPY", "PRDM1", "PRF1", "PRG4",
    "PROX1", "PTGDS", "PTN", "PTPRC", "PVALB", "RAMP2", "RAPGEF3", "RARRES2", "RBM3", "RBP5", "RERGL",
    "RETN", "RGS16", "RHOA", "RIDA", "RNASE1", "RND1", "RTKN2", "RTN4", "S100A1", "S100A12", "SCGB2A1",
    "SCGN", "SEC61B", "SEC62", "SELE", "SELL", "SEMA3C", "SERPINB1", "SERPINB2", "SERPINB3", "SERPINB9",
    "SERPING1", "SFRP2", "SFRP4", "SFTA2", "SH2D3C", "SKP1", "SLAMF1", "SLAMF7", "SLC18A2", "SLC22A8",
    "SLC25A3", "SLC26A2", "SLC26A3", "SLC4A1", "SMIM24", "SMYD2", "SNAI1", "SNCA", "SNCG", "SNTN",
    "SOX17", "SOX18", "SOX2", "SPDEF", "SPI1", "SPIB", "SPON2", "SRPX", "SSR2", "SSR3", "SSR4", "SST",
    "STC1", "STC2", "STEAP4", "SUMO2", "TAC1", "TAT", "TBX3", "TCF15", "TCF4", "TCIM", "TCL1A",
    "TENT5C", "TFF2", "TFPI", "THAP2", "THBS2", "THY1", "TIMP4", "TM4SF18", "TM4SF4", "TMA7", "TMBIM6",
    "TMC5", "TMEM100", "TMEM174", "TMEM52B", "TNC", "TNFRSF13B", "TNFRSF17", "TNFRSF9", "TOMM7",
    "TOP2A", "TRAC", "TRDN", "TREM2", "TSPAN19", "UBE2C", "UGP2", "UMOD", "UPK3B", "UQCC2", "VAMP8",
    "VCAN", "VSIG4", "VTN", "VWA5A", "VWF", "YAF2"
]


XENIUM_MESENCHYMAL_GENES = ['ACTA2','DSP','SNAI1','SUMO2']
XENIUM_EPITHELIAL_GENES = ['EPCAM']
XENIUM_EMT_GENES = XENIUM_MESENCHYMAL_GENES + XENIUM_EPITHELIAL_GENES # all mesenchymal genes
# Cell Type Statistics:
# Macrophages: 28548 / 14.9%
# Fibroblasts: 27535 / 14.4%
# Metaplastic Cells: 19476 / 10.2%
# Tumor Cells: 18847 / 9.9%
# Acinar: 18685 / 9.8%
# CFTR- Tumor Cells: 18592 / 9.7%
# T Cells: 12705 / 6.7%
# Endothelial: 10331 / 5.4%
# Endocrine 1: 9232 / 4.8%
# Endocrine 2: 9185 / 4.8%
# Ductal: 4562 / 2.4%
# CXCL9/10 Cells: 3440 / 1.8%
# B Cells: 3222 / 1.7%
# Mast Cells: 2959 / 1.5%
# Smooth Muscle Cells: 1903 / 1.0%
# Lymphatic Endothelial Cells: 1740 / 0.9%