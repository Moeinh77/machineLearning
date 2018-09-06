from sklearn import tree
import graphviz

toucable_weight = [[1, 430],
                   [1, 350],
                   [1, 250],
                   [1, 250],
                   [1, 150],
                   [1, 120],
                   [1, 110],
                   [1, 170],
                   [1, 180],
                   [1, 260],
                   [1, 450],
                   [1, 320],
                   [1, 900],
                   [1, 2500],
                   [0, 2500],
                   [0, 2200],
                   [0, 2300],
                   [0, 3500],
                   [0, 4500],
                   [0, 2500],
                   [1, 5000],
                   [0, 2700],
                   [0, 3220],
                   [0, 4010]]

label = [1,1,1,1,1,1,1,1,0,1,1,0,1, 0, 0, 1, 1,0,0,0,0,0,0,0]

dtree = tree.DecisionTreeClassifier()

dtree = dtree.fit(toucable_weight, label)


x = input('enter touch availability :')
y = input('enter weight :')


dot_data = tree.export_graphviz(dtree, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("phone decision")
dot_data = tree.export_graphviz(dtree, out_file=None,
                                feature_names=['touch','weight'],
                               # class_names=iris.target_names,
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)

print('#1=phone #0=laptop')

if(dtree.predict([[x,y]])==1):
    print('phone')
else:
    print('laptop')
