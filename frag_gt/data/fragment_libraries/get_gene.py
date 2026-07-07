import pickle
import sys

def count_nested_fragments(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # 提取 Element 1 中的核心字典
        gene_types_dict = data[1].get('gene_types', {})
        
        total_fragments = 0
        all_smiles = set()
        
        print(f"File: {file_path}")
        print(f"Scheme/Version: {data[0]}")
        print(f"Number of Connection Types: {len(gene_types_dict)}")
        
        for conn_type, fragments in gene_types_dict.items():
            # fragments 通常是一个列表，里面存的是片段的 SMILES 或对象
            if isinstance(fragments, (list, set)):
                total_fragments += len(fragments)
                for f in fragments:
                    all_smiles.add(str(f))
            elif isinstance(fragments, dict):
                total_fragments += len(fragments)
                for f in fragments.keys():
                    all_smiles.add(str(f))

        print("-" * 30)
        print(f"Total entries across all types: {total_fragments}")
        print(f"Unique Fragment SMILES (Genes): {len(all_smiles)}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        count_nested_fragments(sys.argv[1])
