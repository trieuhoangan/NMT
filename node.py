import copy
from preprocess import load_token_list_from_file
import numpy as np
class Node():
    def __init__(self,id=-1,text='',level=-1):
        self.id = id
        self.text = text
        self.level = level
        self.childList = []
        self.childNum = 0
        self.father = None
        
    def addChild(self,child):
        self.childList.append(child)
        self.childNum = self.childNum +1


    def print_all(self):
        if self.text != "":
            print("id {} text {} at level {}".format(self.id,self.text,self.level))
        else:
            print("*****")
        for child in self.childList:
            child.print_all()
    def print_all_child(self):
        for child in self.childList:
            child.print_all()
    
    def removeChild(self,child):
        if child in self.childList:
            self.childList.remove(child)
            child.father = None
            self.childNum = self.childNum - 1
    def count_leaf(self):
        if self.childNum == 0:
            return 1
        else:
            count = 0
            for child in self.childList:
                count = count + child.count_leaf()
            return count
    def count_false_node(self):
        
        if self.text != "" and self.childNum== 0:
            return 0
        else:
            count = 0
            if self.text != "" and self.childNum!= 0:
                count += 1
            for child in self.childList:
                count = count + child.count_false_node()
            return count    
    def count_node(self):
        if self.childNum == 0:
            return 1
        else:
            count = 1
            for child in self.childList:
                count = count + child.count_node()
            return count
class Tree():
    '''
        generate tree from dependancy level
    '''
    def __init__(self,nodelist):
        self.nodeList = nodelist

    def get_level_list(self,List):
        levels = []
        for node in List:
            if node.level not in levels:
                levels.append(node.level)
        return levels

    def get_list_of_node_by_level(self,List,level):
        nodes = []
        for node in List:
            if node.level == level:
                nodes.append(node)
        return nodes

    def generate_tree(self):
        
        sorted_list = self.sort_as_level()
        levels = self.get_level_list(sorted_list)
        nodeNum = len(sorted_list)
        for i in range(nodeNum-1,1,-1):
            current_level = sorted_list[i].level
            level_index = levels.index(current_level)
            father_level = levels[level_index-1]
            fathers = self.get_list_of_node_by_level(sorted_list,father_level)
            min_dist,father_id = self.cal_node_dist(fathers,sorted_list[i])
            for j in range(0,nodeNum):
                if sorted_list[j].id == father_id:
                    sorted_list[j].addChild(sorted_list[i])
                    sorted_list[i].father = sorted_list[j]
                    break
        return sorted_list[0]

    def cal_node_dist(self,nodeList,childnode):
        min_dist = 10000
        father = Node()
        for node in nodeList:
            dist = node.id - childnode.id
            if dist <0:
                dist = dist*-1
            if dist < min_dist:
                min_dist = dist
                father = node
        return min_dist,father.id

    def sort_as_level(self):
        listNode = copy.deepcopy(self.nodeList)
        numNode = len(listNode)
        for i in range(0,numNode):
            for j in range(i+1,numNode):
                if listNode[i].level > listNode[j].level:
                    tmpNode = copy.deepcopy(listNode[j])
                    listNode[j] = copy.deepcopy(listNode[i])
                    listNode[i] = copy.deepcopy(tmpNode)
        return listNode
    
    def up_node(self,node):
        new_node = Node()
        if node.text != "":
            new_node.addChild(node)
            node.father = new_node
            if node.childNum > 1:
                new_child = Node()
                for child in node.childList:
                    new_child.addChild(child)
                    child.father = new_child
                new_child.father = new_node
                new_node.addChild(new_child)
            else:
                new_node.addChild(node.childList[0])
                node.childList[0].father = new_node
            node.childList=[]
            node.childNum = 0
        else:
            new_node.addChild(node.childList[0])
            node.removeChild(node.childList[0])
            new_node.childList[0].father = new_node
            new_node.addChild(node)
            node.father = new_node
        
        return new_node
    
    def create_bin_tree(self,root):
        new_root = self.up_node(root)
        tmp = new_root
        list_cross_node = [new_root.childList[1]]
        while len(list_cross_node)!=0:
            tmp2 = list_cross_node[-1]
            tmp = tmp2.father
            list_cross_node.remove(tmp2)

            if tmp2.childNum > 2:
                tmp.removeChild(tmp2)
                tmp2 = self.up_node(tmp2)
                tmp.addChild(tmp2)
                tmp2.father = tmp

            if tmp2.text != "" and tmp2.childNum != 0:
                tmp.removeChild(tmp2)
                tmp2 = self.up_node(tmp2)
                tmp.addChild(tmp2)

            list_cross_node.extend(tmp2.childList)
 
        return new_root
    def generate_adjency_list(self,root):
        adjency_list = []
        awaiting_list = [root]
        parsed_list = []
        while len(awaiting_list) != 0:
            tmp = awaiting_list[0]
            awaiting_list.remove(tmp)
            parsed_list.append(tmp)
            awaiting_list.extend(tmp.childList)
        numNode = len(parsed_list)
        for i in range(0,numNode):
            listi = []
            for j in range(0,numNode):
                if parsed_list[j].father == parsed_list[i]:
                    listi.append(j)
                    continue
            adjency_list.append((parsed_list[i].text,listi))
        return adjency_list

def load_token_list_from_file(file_path):
  lines =  open(file_path,'r',encoding='utf-8').read().split('\n')
  token_list = []
  for line in lines:
    if line =='':
      continue
    if line =='None':
      token_list.append([])
      continue
    words = []
    head_indexes = []
    feature = line.split('|')
    for word in feature[0].split(','):
      if word !='':
        words.append(word)
    for index in feature[3].split(','):
      if index!='':
        head_indexes.append(index.split(':'))
    listone = []
    listone.append([words])
    listone.append([head_indexes])
    token_list.append(listone)
  return token_list
def create_node_list(token_list):
    nodeListes = []
    for tokens in token_list:
        nodes = []
        num_word = len(tokens[0][0])
        num_depen = len(tokens[1][0])
        
        if num_word == num_depen:
            for i in range(0,num_word):
                node = Node(id=i,text=tokens[0][0][i],level=int(tokens[1][0][i][0]))
                nodes.append(node)
        else:
            limit = min(num_depen,num_word)
            for i in range(0,limit):
                node = Node(id=i,text=tokens[0][0][i],level=int(tokens[1][0][i][0]))
                nodes.append(node)     
        nodeListes.append(nodes)
    return nodeListes
if __name__ == "__main__":
    tokenList = load_token_list_from_file("data/dev_phonlp_token_list.txt")
    nodeList = create_node_list(tokenList)

    tree = Tree(nodeList[0])
    
    root = tree.generate_tree()
    root = tree.create_bin_tree(root)
    # root.print_all()
    # print(root.count_node())
    adjency_list = tree.generate_adjency_list(root)
    print(adjency_list)