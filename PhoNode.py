
class PhoNode:
  def __init__(self,text='',pos='',ner='',id=-1,level=-1,dependency_relation=''):
    self.text = text
    self.pos = pos
    self.ner = ner
    self.id = id
    self.level = level
    self.dependency_relation = dependency_relation
    self.childList = []
    self.father = None
    self.isLeaf = False
    self.part = []
    self.left = None
    self.right = None
    self.h = None
    self.c = None
    self.word_index = 0
  def setRight(self,listnode):
    self.right = listnode

  def setLeft(self, listnode):
    self.left = listnode

  def setPart(self,parts):
    self.part = parts

  def getLevel(self):
    return self.level

  def getID(self):
    return self.id

  def addChild(self,node):
    self.childList.append(node)
    
  def addFather(self,node):
    self.father = node

  def levelDown(self):
    self.level = self.level +1
    for child in self.childList:
      child.levelDown()

  def printChildList(self):
    if len(self.childList)>0:
      print('node number {}'.format(self.id))
      for child in self.childList:
        child.print_out()
      for child in self.childList:
        child.printChildList()
    else:
      print("node {} is a leaf at level {}".format(self.id,self.level))

  def simplifiedNode(self):
    if len(self.childList) >2:
      return 0
    return -1

  def checkLeaf(self):
    if len(self.childList) ==0:
      self.isLeaf = True
    else:
      self.isLeaf = False

  def print_out_bin_tree(self):
    text = ''
    if len(self.part) != 0:
      for node in self.part:
        text = text + ' ' + node.text
    
    print('{}'.format(text))
    # if self.left!=None:
    #   self.left.print_out_bin_tree()
    # if self.right!=None:
    #   self.right.print_out_bin_tree()
    # left_text = ''
    # right_text = ''
    # if self.left != None:
    #   for node in self.left.part:
    #     left_text = left_text +' '+node.text
    # if self.right!=None:
    #   for node in self.right.part:
    #     right_text = right_text + ' ' + node.text
    # print('main: {} |  left : {}  | right : {}'.format(text,left_text,right_text))
    if self.left!=None:
      self.left.print_out_bin_tree()
    if self.right!=None:
      self.right.print_out_bin_tree()

  def clear_bin_tree(self):
    self.clear_part_context()
    if self.left!=None:
      self.left.clear_bin_tree()
    if self.right!=None:
      self.right.clear_bin_tree()

  def clear_part_context(self):
    if len(self.part) > 1:
      self.part = []

  def convert_bin_tree_to_word_index(self,model):
    if len(self.part)!=0:
      if self.part[0].text in model.vocab:
        self.part[0].word_index = model.vocab[self.part[0].text].index
      else:
        self.part[0].word_index = -1
    if self.left!=None:
      self.left.convert_bin_tree_to_word_index(model)
    if self.right!=None:
      self.right.convert_bin_tree_to_word_index(model)

  def print_word_indices(self):
    if len(self.part) !=0:
      print('index',self.part[0].word_index)
    if self.left!=None:
      self.left.print_word_indices()
    if self.right!=None:
      self.right.print_word_indices()

  def print_out(self):
    print("id: {}, text: {} , pos: {} , ner : {}, level: {} , dependency_relation: {}".format(self.id,self.text,self.pos,self.ner,self.level,self.dependency_relation))


class Tree_:

  def __init__(self,token_list):
    self.nodeList = self.make_node_list(token_list)
    self.root = PhoNode('','','',-1,'-1','')
    self.root = self.getRoot()
    self.numNode = len(token_list[0][0])
    # self.create_connection()

  def make_node_list(self,token_list):
    
    num_token = len(token_list[3][0])
    nodeList = []
    for i in range(num_token):
      idx = i
      texts = token_list[0][0][i]
      # pos = token_list[1][0][i][0]
      # ner = token_list[2][0][i]
      levels = token_list[3][0][i][0]
      dependency_relations = token_list[3][0][i][1]
      node = PhoNode(text=texts,id=idx,level = int(levels),dependency_relation=dependency_relations)
      # node.print_out()
      nodeList.append(node)
    return nodeList

  def getRoot(self):
    node_ = None
    for node in self.nodeList:
      if node.getLevel()==0:
        node_ = node
        return node_
    return PhoNode('','','',-1,-1,'')

  def create_connection(self):
    max_level = self.get_max_level()
    self.nodeList = self.sort_as_level()
    levelList = self.create_level_list(self.nodeList)
    numLevel = len(levelList)
    for i in range(numLevel):
      if i ==0:
        continue
      else:
        nodeNum = len(levelList[i])
        for j in range(nodeNum):
          mindist,father = self.cal_node_dist(levelList[i-1],levelList[i][j])
          father.addChild(levelList[i][j])
          levelList[i][j].addFather(father)

  def get_min_level_token(self,nodelist):
    min_level = 1000
    next_root = -1
    for node in nodelist:
      if node.level < min_level:
        min_level = node.level
        next_root = node.id
    return next_root,min_level

  def make_binary_tree(self,parts):
    root = PhoNode()
    root.setPart(parts)
    if len(parts) == 1:
      return root
    if len(parts) == 2:
      # print('got 2 word alone')
      # parts[0].print_out()
      # parts[1].print_out()
      leftNode = PhoNode()
      rightNode = PhoNode()
      leftNode.setPart([parts[0]])
      rightNode.setPart([parts[1]])
      root.setLeft(leftNode)
      root.setRight(rightNode)
      leftNode.father = root
      rightNode.father = root
      return root
    id_min,min_lv = self.get_min_level_token(parts)
    left_part = []
    right_part = []
    center_node = None
    for node in parts:
      if node.id < id_min:
        left_part.append(node)
      elif node.id > id_min:
        right_part.append(node)
      elif node.id == id_min and center_node == None:
        center_node = node
      else:
        right_part.append(node)
    if len(left_part)!=0 and len(right_part)!=0:
      left_part.append(center_node)
      right = self.make_binary_tree(right_part)
      right.father = root
      root.setRight(right)
      left = self.make_binary_tree(left_part)
      left.father = root
      root.setLeft(left)
    elif len(left_part) == 0:
      center = PhoNode()
      center.setPart([center_node])
      root.setLeft(center)
      center.father = root
      right = self.make_binary_tree(right_part)
      right.father = root
      root.setRight(right)
    elif len(right_part)== 0:
      center = PhoNode()
      center.setPart([center_node])
      root.setRight(center)
      center.father = root
      left = self.make_binary_tree(left_part)
      left.father = root
      root.setLeft(left)
    return root

  def cal_node_dist(self,nodeList,childnode):
    min_dist = 10000
    father = childnode
    for node in nodeList:
      dist = node.id - childnode.id
      if dist <0:
        dist = dist*-1
      if dist < min_dist:
        min_dist = dist
        father = node
    return min_dist,father

  def create_level_list(self,sortedList):
    current_level = 0
    levelList = []
    currentLevel = []
    for node in sortedList:
      if node.level == current_level:
        currentLevel.append(node)
      else:
        current_level = node.level
        levelList.append(currentLevel)
        currentLevel=[]
        currentLevel.append(node)
    # for level in levelList:
    #   level[0].print_out()
    return levelList

  def sort_as_level(self):
    listNode = self.nodeList
    numNode = self.numNode
    for i in range(0,numNode):
      for j in range(i+1,numNode):
        if listNode[i].level > listNode[j].level:
          tmpNode = listNode[j]
          listNode[j] = listNode[i]
          listNode[i] = tmpNode
    # for node in listNode:
    #   node.print_out()
    return listNode
  
  def get_max_level(self):
    max_level = 0
    for node in self.nodeList:
      if node.level > max_level:
        max_level = node.level
    return max_level

  def print_out(self):
    for node in self.nodeList:
      node.print_out()