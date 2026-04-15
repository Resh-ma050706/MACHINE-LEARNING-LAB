print("S.RESHMA 24BAD097")
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
file_path = r"C:\Users\ASUS\Downloads\exp 8.1\Groceries_dataset.csv"
df = pd.read_csv(file_path)
print("First 5 rows:")
print(df.head())
transactions = df.groupby(['Member_number', 'Date'])['itemDescription'] \
                 .apply(list).values.tolist()
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)
frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
rules = rules[rules['lift'] > 1]
if rules.empty:
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
top_items = frequent_itemsets.sort_values(by='support', ascending=False).head(10)
plt.figure()
plt.bar(range(len(top_items)), top_items['support'])
plt.xticks(range(len(top_items)), [str(i) for i in top_items['itemsets']], rotation=45)
plt.title("Top Frequent Itemsets")
plt.xlabel("Itemsets")
plt.ylabel("Support")
plt.show()
if not rules.empty:
    plt.figure()
    plt.scatter(rules['support'], rules['confidence'])
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Support vs Confidence")
    plt.grid()
    plt.show()
plt.figure(figsize=(10,7))
G = nx.DiGraph()
for _, row in rules.head(10).iterrows():
    for ant in row['antecedents']:
        for con in row['consequents']:
            G.add_edge(ant, con, weight=row['confidence'])
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=8)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Association Rules Network Graph")
plt.show()
print("\nINTERPRETATION:")
for index, row in rules.head(5).iterrows():
    print(f"If a customer buys {list(row['antecedents'])}, "
          f"they are likely to buy {list(row['consequents'])} "
          f"(Confidence: {row['confidence']:.2f}, Lift: {row['lift']:.2f})")