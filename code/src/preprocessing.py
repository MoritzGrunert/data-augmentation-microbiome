from biom import load_table
import pandas as pd

table = load_table("/Users/moritzgrunert/Desktop/Projects/2.Seminar/data_augmentation_microbiome/code/data/441_otu_table.biom")
data = table.to_dataframe(dense=False).T

# remove OTUs with total counts across all samples < 3
data = data[data.sum(axis=1) > 3]

# remove samples with < 10,000 sequences  (=> keep >= 10,000)
depth = data.sum(axis=1)  
high_ids = depth.index[depth > 10_000]
filtered_data = data.loc[high_ids]

meta_data = pd.read_csv(
    "/Users/moritzgrunert/Desktop/Projects/2.Seminar/data_augmentation_microbiome/code/data/meta_data.txt",  
    sep="\t"              
)

meta_data = meta_data.set_index("sample_name")
print(f"Meta data shape: {meta_data.shape}")
print(f"Meta data columns: {meta_data.columns} \n")


# filter out samples from people on antibiotics
# antibiotics is boolean with NaN -> keep only explicit False
print(f"Meta data antibiotics unique values: {meta_data['antibiotics'].unique()}")
print(meta_data["antibiotics"].value_counts(), "\n")
meta_data = meta_data[meta_data["antibiotics"] == False]

# Create binary diagnosis column 
print(f"Meta data diagnosis unique values: {meta_data['diagnosis'].unique()}")
print(meta_data["diagnosis"].value_counts(), "\n")
mapping = {"control": True, "no": True, "CD": False, "UC": False, "IC": False}
meta_data["diagnosis_binary"] = meta_data["diagnosis"].map(mapping)
print(meta_data["diagnosis_binary"].value_counts(), "\n")

# keep only rows with mapped labels (should be all, but safe)
meta_data = meta_data["diagnosis_binary"].dropna()

# merge filtered data with meta data
merged_data = filtered_data.join(meta_data, how="inner")
merged_data.shape
print(merged_data["diagnosis_binary"].value_counts(), "\n")
print(f"Merged data shape: {merged_data.shape} \n")

merged_data.to_csv("/Users/moritzgrunert/Desktop/Projects/2.Seminar/data_augmentation_microbiome/code/data/filtered_data.csv")