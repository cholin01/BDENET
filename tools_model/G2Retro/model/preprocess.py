import sys, os
import pickle
import rdkit
import time
import pandas as pd
import pdb
import multiprocessing as mp
from functools import partial
from mol_tree import MolTree, update_revise_atoms, get_synthon_trees
from chemutils import get_idx_from_mapnum, copy_edit_mol
from multiprocessing import Pool
from argparse import ArgumentParser
from rdkit import Chem


def build_moltree(data_tuple, use_dfs=True, shuffle=False):
    """ construct molecular trees """
    # Unpack the tuple containing the pandas row and the bde dictionary for this specific molecule
    data, bde_info = data_tuple

    react_moltrees = []
    react_smiles = data['rxn_smiles'].split(">>")[0]
    prod_smiles = data['rxn_smiles'].split(">>")[1]

    # Pass the bde_info into the MolTree initialization
    prod_moltree = MolTree(prod_smiles, use_brics=True, decompose_ring=True, bde_info=bde_info)
    react_moltree = MolTree(react_smiles)
    
    try:
        update_revise_atoms(prod_moltree, react_moltree, use_dfs=use_dfs, shuffle=shuffle)
    except:
        return (None, None, None, set())
    
    vocab = set()
    
    synthon_tree = get_synthon_trees(react_moltree)
    for node in react_moltree.mol_tree.nodes:
        if react_moltree.mol_tree.nodes[node]['revise'] == 1:
            vocab.add(react_moltree.mol_tree.nodes[node]['label'])
    
    if len(synthon_tree.smiles.split(".")) != len(react_moltree.smiles.split(".")):
        return (None, None, None, set())
    return (prod_moltree, synthon_tree, react_moltree, vocab)
    

def get_template(mol_tree1, mol_tree2):
    if len(mol_tree1.ring[0]) == 0:
        product_idxs = list(set([idx for bond in mol_tree1.order for idx in bond[:2] if idx != -1]))
    else:
        product_idxs = mol_tree1.ring[0]
    
    pro_temp_smiles = Chem.MolFragmentToSmiles(mol_tree1.mol, product_idxs, kekuleSmiles=True)
    mol = Chem.MolFromSmiles(pro_temp_smiles, sanitize=False)
    new_mol = copy_edit_mol(mol).GetMol()
    new_mol = Chem.RemoveHs(new_mol)
    for atom in new_mol.GetAtoms():
        atom.SetAtomMapNum(0)
        atom.SetNumExplicitHs(0)
    pro_temp_smiles = Chem.MolToSmiles(new_mol)
    atom_orders = [int(num) for num in new_mol.GetProp('_smilesAtomOutputOrder')[1:-1].split(",") if len(num) > 0]
    
    mapnum_mapnum_dict = {}
    for i, num in enumerate(atom_orders):
        new_mol.GetAtomWithIdx(num).SetAtomMapNum(0)
        mol_tree1.mol_graph.nodes[num]['atommap'] = i+1
        mapnum_mapnum_dict[mol.GetAtomWithIdx(num).GetAtomMapNum()] = i+1
    
    pro_temp_smiles = Chem.MolToSmiles(new_mol)
    
    product_nums = [mol_tree1.mol_graph.nodes[idx]['idx'] for idx in product_idxs]
    
    map_dict = get_idx_from_mapnum(mol_tree2.mol)
    react_idxs = [map_dict[num] for num in product_nums]
    react_temp_smiles = Chem.MolFragmentToSmiles(mol_tree2.mol, react_idxs, kekuleSmiles=True)
    mol = Chem.MolFromSmiles(react_temp_smiles, sanitize=False)
    new_mol = copy_edit_mol(mol).GetMol()
    new_mol = Chem.RemoveHs(new_mol)
    for atom in new_mol.GetAtoms():
        atom.SetAtomMapNum(0)
        atom.SetNumExplicitHs(0)
    react_temp_smiles = Chem.MolToSmiles(new_mol)
    
    mol_tree1.template = [pro_temp_smiles, react_temp_smiles]
    
    return pro_temp_smiles, react_temp_smiles


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = ArgumentParser()
    parser.add_argument('--train', type=str, default="/home/gouqiaolin/model/G2Retro/data/bde_train/filtered_train.csv",
                        help="specify the name of file with training data to be processed")
    parser.add_argument('--path', type=str, default="/home/gouqiaolin/model/G2Retro/data/bde_train",
                        help="specify the name of file with training data to be processed")
    parser.add_argument('--output', type=str, default="train_tensors_bde", help="specify the name of processed dataset.")
    parser.add_argument('--use_bfs', action="store_true")
    parser.add_argument('--shuffle', action="store_true")
    parser.add_argument('--use_class', action="store_false", help="whether add class into the dataset")
    parser.add_argument('--ncpu', type=int, default=10, help="specify the number of CPUs used for preprocessing.")
    # Add argument for the BDE database
    parser.add_argument('--bde_data', type=str, default="/home/gouqiaolin/model/G2Retro/data/bde_train/react_prod.pkl", help="Path to BDE dictionary")
    args = parser.parse_args()

    # Load the BDE database into memory
    print("Loading BDE database...")
    bde_database = {}
    if os.path.exists(args.bde_data):
        with open(args.bde_data, 'rb') as f:
            bde_database = pickle.load(f)
    else:
        print(f"Warning: BDE database {args.bde_data} not found. Features will default to 0.0")

    cpu_count = mp.cpu_count()
    print("start...")
    pdata = []
    path = args.path
    all_data = pd.read_csv(args.train, sep=',')

    # Package the pandas row with its specific BDE dictionary lookup
    all_data_list = []
    for i in range(len(all_data)):
        row = all_data.iloc[i, :]
        prod_smi = row['rxn_smiles'].split(">>")[1]
        mol_bde_dict = bde_database.get(prod_smi, {})
        all_data_list.append((row, mol_bde_dict))

    func = partial(build_moltree, use_dfs=not args.use_bfs, shuffle=args.shuffle)
    with Pool(processes=cpu_count) as pool:
        mol_trees = pool.map(func, all_data_list)
    
    templates = {}
    for product_tree, _, react_tree, _ in mol_trees:
        if product_tree is None or len(product_tree.order) == 1: continue
        template = get_template(product_tree, react_tree)
        if template not in templates: templates[template] = []
        templates[template].append((product_tree.smiles, react_tree.smiles))
    
    template_count = {}
    template_types = {}
    for template in templates:
        template_count[template] = len(templates[template])
        
        if template_count[template] < 5: continue
        if template[0] not in template_types:
            template_types[template[0]] = []
        template_types[template[0]].append(template[1])
     
    f = open("%s/template.txt" % (path), 'w')
    for prod, reacts in template_types.items(): f.write("%s %s\n" % (prod, ",".join(reacts)))
    f.close()
    
    vocab = list(set([label for mol_tree in mol_trees for label in mol_tree[3]]))
    if not os.path.exists("%s/vocab.txt" % (path)): 
        f = open("%s/vocab.txt" % (path), 'w')
        for word in vocab: f.write("%s\n" % (word))
        f.close()

    new_mol_trees = []
    # 🌟 修改点 1：将变量名临时改为 data_tuple 以明确其结构
    for mol_tree, data_tuple in zip(mol_trees, all_data_list):
        # 🌟 修改点 2：从元组中解包出原始的 pandas 数据行
        data = data_tuple[0]

        if mol_tree[0] is None:
            print("%s,cannot be processed" % (data['rxn_smiles']))
        else:
            # 这里继续使用解包出来的 data，保证后续代码正常运行
            new_mol_trees.append((mol_tree[:3], data))

    mol_trees = new_mol_trees
    
    removed_idxs = []
    for i in range(len(mol_trees)):
        mol_tree, data = mol_trees[i]
        prod_moltree, synthon_mol_tree, react_moltree = mol_tree
        
        num_deletes, num_changes = 0, 0
        visited = []
        
        for edge in prod_moltree.mol_graph.edges:
            if (edge[1], edge[0]) not in visited:
                visited.append( (edge[0], edge[1]) )
            else:
                continue
            if 'delete' in prod_moltree.mol_graph[edge[0]][edge[1]] and prod_moltree.mol_graph[edge[0]][edge[1]]['delete'] == 1:
                num_deletes += 1
            
            if 'change' in prod_moltree.mol_graph[edge[0]][edge[1]] and prod_moltree.mol_graph[edge[0]][edge[1]]['change'] >= 0:
                num_changes += 1
        
        if num_deletes > 1 and len(prod_moltree.ring[0]) == 0:
            removed_idxs.append(i)
            print("%s,multi deletion (not ring),%d" % (data['rxn_smiles'], num_deletes))
            continue
            
        if num_changes > 1 and len(prod_moltree.ring[0]) == 0:
            removed_idxs.append(i)
            print("%s,multi changes (not ring),%d" % (data['rxn_smiles'], num_changes))
            continue
        
        if len(prod_moltree.ring[0]) > 0:
                removed_idxs.append(i)
                print("%s,new ring,%s,%s" % (data['rxn_smiles'], prod_moltree.template[0], prod_moltree.template[1]))
                continue
        
        num_attach = 0
        for node in prod_moltree.mol_graph.nodes:
            if 'attach' in prod_moltree.mol_graph.nodes[node] and prod_moltree.mol_graph.nodes[node]['attach'] == 1:
                num_attach += 1
        
        if num_deletes == 0 and num_changes == 0 and num_attach >= 2:
            removed_idxs.append(i)
            print("%s,multi attachs,%d" % (data['rxn_smiles'], num_attach))
            continue
        
        if_break = False
        for bond in react_moltree.mol_graph.edges():
            if react_moltree.mol_graph[bond[0]][bond[1]]['label'] < 0: continue
            if react_moltree.mol.GetBondBetweenAtoms(bond[0], bond[1]).IsInRing():
                num1 = react_moltree.mol.GetAtomWithIdx(bond[0]).GetAtomMapNum()
                num2 = react_moltree.mol.GetAtomWithIdx(bond[0]).GetAtomMapNum()
          
                if num1 + num2 == abs(num1 - num2) and num1 + num2 != 0:
                    removed_idxs.append(i)
                    if_break = True
                    print("%s,build new ring,0" % (data['rxn_smiles']))
                    break
            if if_break: break
    
    if args.use_class:
        selected_mol_trees = [(mol_trees[i][0], mol_trees[i][1][0]) for i in range(len(mol_trees)) if i not in removed_idxs]
    else:
        selected_mol_trees = [(mol_trees[i][0], None) for i in range(len(mol_trees)) if i not in removed_idxs]
    
    with open("%s/%s.pkl" % (path, args.output), 'wb') as f:
        pickle.dump(selected_mol_trees, f, pickle.HIGHEST_PROTOCOL)
