#!/usr/local/bin/python3

"""
This is where you should write your AI code!

Authors: [Harsha Raja Shivakumar  | Maithreyi Manur Narasimha Prabhu | Sunny Bhati]

Based on skeleton code by Abhilash Kuhikar, October 2019
"""

from logic_IJK import Game_IJK
from copy import deepcopy
import copy
import random

# Suggests next move to be played by the current player given the current game
#
# inputs:
#     game : Current state of the game 
#
# This function should analyze the current state of the game and determine the 
# best move for the current player. It should then call "yield" on that move.

MAX, MIN = 1000, -1000 
  
# Returns optimal value for current player  
#(Initially called for root and maximizer)  
index = -1

class Game_IJK_new:
    def __init__(self, game, currentPlayer, deterministic):
        self.__game = game
        self.__current_player = +1 if currentPlayer == '+' else -1
        self.__previous_game = self.__game
        self.__new_piece_loc = (0,0)
        self.__deterministic = deterministic

    def __switch(self):
        self.__current_player = -self.__current_player
    
    def isGameFull(self):
        for i in range(len(self.__game)):
            for j in range(len(self.__game[0])):
                if self.__game[i][j] == ' ':
                    return False
        return True
    
    def __game_state(self,mat):
        highest = {'+': 'A', '-': 'a'}
        
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                if (mat[i][j]).isupper():
                    highest['+'] = chr(max(ord(mat[i][j]), ord(highest['+'])))
                if (mat[i][j]).islower():
                    highest['-'] = chr(max(ord(mat[i][j]), ord(highest['-'])))
        
        if highest['+'] == 'K' or highest['-'] == 'k' or self.isGameFull():
            if highest['+'].lower() != highest['-']:
                return highest['+'] if highest['+'].lower()>highest['-'] else highest['-']
            return 'Tie'

        return 0

    def __reverse(self,mat):
        new = []
        for i in range(len(mat)):
            new.append([])
            for j in range(len(mat[0])):
                new[i].append(mat[i][len(mat[0])-j-1])
        return new
    
    def __transpose(self,mat):
        new = []
        for i in range(len(mat[0])):
            new.append([])
            for j in range(len(mat)):
                new[i].append(mat[j][i])
        return new
    
    def __cover_up(self,mat):
        new = [[' ' for _ in range(len(self.__game))]for _ in range(len(self.__game))]

        done = False
        for i in range(len(self.__game)):
            count = 0
            for j in range(len(self.__game)):
                if mat[i][j] != ' ':
                    new[i][count] = mat[i][j]
                    if j != count:
                        done = True
                    count += 1
        return (new, done)
    
    def __merge(self,mat):
        global current_player

        done = False
        for i in range(len(self.__game)):
            for j in range(len(self.__game)-1):
                if mat[i][j] == mat[i][j+1] and mat[i][j] != ' ':
                    mat[i][j] = chr(ord(mat[i][j])+ 1)
                    mat[i][j+1] = ' '
                    done = True
                elif mat[i][j].upper() == mat[i][j+1].upper() and mat[i][j] != ' ':
                    mat[i][j] = chr(ord(mat[i][j])+ 1)
                    mat[i][j] = mat[i][j].upper() if self.__current_player > 0 else mat[i][j].lower()
                    mat[i][j+1] = ' '
                    done = True
        return (mat, done)
    
    def __up(self,game):
        #print("up")
        # return matrix after shifting up
        game = self.__transpose(game)
        game, done = self.__cover_up(game)
        temp = self.__merge(game)
        game = temp[0]
        done = done or temp[1]
        game = self.__cover_up(game)[0]
        game = self.__transpose(game)
        if done == True:
            self.__game = copy.deepcopy(game)
        return (game, done)
    
    def __down(self,game):
        #print("down")
        game = self.__reverse(self.__transpose(game))
        game, done = self.__cover_up(game)
        temp = self.__merge(game)
        game = temp[0]
        done = done or temp[1]
        game = self.__cover_up(game)[0]
        game = self.__transpose(self.__reverse(game))
        if done == True:
            self.__game = copy.deepcopy(game)
        return (game, done)
    
    def __left(self,game):
        #print("left")
        # return matrix after shifting left
        game, done = self.__cover_up(game)
        temp = self.__merge(game)
        game = temp[0]
        done = done or temp[1]
        game = self.__cover_up(game)[0]
        if done == True:
            self.__game = copy.deepcopy(game)
        return (game, done)
    
    def __right(self,game):
        #print("right")
        # return matrix after shifting right
        game = self.__reverse(game)
        game, done = self.__cover_up(game)
        temp = self.__merge(game)
        game = temp[0]
        done = done or temp[1]
        game = self.__cover_up(game)[0]
        game = self.__reverse(game)
        if done == True:
            self.__game = copy.deepcopy(game)
        return (game, done)
    
    def __skip(self):
        x, y = self.__new_piece_loc
        self.__game[x][y] = self.__game[x][y].swapcase()
    '''
    Expose this method to client to print the current state of the board
    '''
    def printGame(self):
        str_game = [['______' for _ in range(len(self.__game))] for _ in range(len(self.__game))]
        
        for i in range(len(self.__game)):
            for j in range(len(self.__game)):
                str_game[i][j] = "_"+self.__game[i][j]+"_"
        
        for i in range(len(self.__game)):
            print("|".join(str_game[i]))
        print("\n")

    def __add_piece(self):
        if self.__deterministic:
            for i in range(len(self.__game)):
                for j in range(len(self.__game)):
                    if self.__game[i][j] == ' ':
                        self.__game[i][j] = 'A' if self.__current_player>0 else 'a'
                        self.__new_piece_loc = (i,j)
                        return
        else:
            open=[]
            for i in range(len(self.__game)):
                for j in range(len(self.__game)):
                    if self.__game[i][j] == ' ':
                        open += [(i,j),]

            if len(open) > 0:
                r = random.choice(open)
                self.__game[r[0]][r[1]] = 'A' if self.__current_player>0 else 'a'
                self.__new_piece_loc = r

                
    def makeMove(self,move):
        if move not in ['U','L','D','R']:
            raise InvalidMoveException

        self.__previous_game = self.__game
        if move == 'L':
            self.__left(self.__game)
        if move == 'R':
            self.__right(self.__game)
        if move == 'D':
            self.__down(self.__game)
        if move == 'U':
            self.__up(self.__game)
        if move == 'S':
            self.__skip()
        
        '''
        Switch player after the move is done
        '''
        self.__switch()
        if move != 'S':
            self.__add_piece()
        #self.printGame()
        
        return copy.deepcopy(self)

    def makeMoveNew(self,move):
        if move not in ['U','L','D','R']:
            raise InvalidMoveException

        self.__previous_game = self.__game
        if move == 'L':
            self.__left(self.__game)
        if move == 'R':
            self.__right(self.__game)
        if move == 'D':
            self.__down(self.__game)
        if move == 'U':
            self.__up(self.__game)
        if move == 'S':
            self.__skip()
        
        '''
        Switch player after the move is done
        '''
        self.__switch()
        
        return copy.deepcopy(self)

    def getDeterministic(self):
        return self.__deterministic
    
    def getGame(self):
        return copy.deepcopy(self.__game)
    
    '''player who will make the next move'''
    def getCurrentPlayer(self):
        return '+' if self.__current_player > 0 else '-'

    ''' '+' : '+' has won
       '-1' : '-' has won
       '' : Game is still on
    '''
    def state(self):
        return self.__game_state(self.__game)

current_index = 0
def minimax(depth, nodeIndex, maximizingPlayer,  
            values, alpha, beta):  
    global index

    if depth == 4:
        return values[nodeIndex],index  
  
    if maximizingPlayer:  
       
        best = MIN 
  
        # Recur for left and right children  
        for i in range(0, 4):  
              
            val,index = minimax(depth + 1, nodeIndex * 4 + i,  
                          False, values, alpha, beta)  
            best = max(best, val)  

            if alpha < best:
                alpha = best
                index = i
  
            # Alpha Beta Pruning  
            if beta <= alpha:  
                break 
           
        return best,index
       
    else: 
        best = MAX 
  
        # Recur for left and  
        # right children  
        for i in range(0, 4):  
           
            val,index = minimax(depth + 1, nodeIndex * 4 + i,  
                            True, values, alpha, beta)  
            best = min(best, val)  
            beta = min(beta, best)  
  
            # Alpha Beta Pruning  
            if beta <= alpha:  
                break 
           
        return best,index

def find_successors(board, player, deterministic):
    moves = ['L', 'R', 'U', 'D']
    child_nodes = []
    for i in moves:
        game1 = Game_IJK(board, player, deterministic)
        child_nodes.append(game1.makeMove(i).getGame())
    return child_nodes

def find_successors1(board1, player1, depth1, deterministic, tree_depth, parent_index):

    global current_index

    moves = ['L', 'R', 'U', 'D']
    child_nodes = []
    for i in moves:
        game1 = Game_IJK_new(board1, player1, deterministic)
        if depth1 == tree_depth-1:
            game1 = game1.makeMoveNew(i)
            booleanValue = game1.isGameFull()
            if booleanValue == False:
                child_nodes.append((game1.getGame(), int(not(player1)), depth1+1, heuristic(game1.getGame(), int(not(player1))), board1, current_index, parent_index))
            current_index += 1
        else:
            game1 = game1.makeMoveNew(i)
            booleanValue = game1.isGameFull()
            if booleanValue == False:
                child_nodes.append((game1.getGame(), int(not(player1)), depth1+1, 0, board1, current_index, parent_index))
            current_index += 1
    return child_nodes

def add_move(board1, player, i, j):
    board2 = deepcopy(board1)
    board2[i][j] = 'A' if player>0 else 'a'
    return board2

def find_chance(board1, player, depth, parent_index):
    global current_index

    board2 = deepcopy(board1)
    result = []
    temp = False
    count = emptyTilesCount(board2)

    for i in range(len(board2)):
        for j in range(len(board2)):
                if board2[i][j] == ' ':
                    temp = True
                    break
        if temp:
            break

    while count >= 1:
        if board1[i][j] == ' ':
            count -= 1
            result += [(add_move(board2, player, i, j), -1 , depth+1, 0, board1, current_index, parent_index)]
            current_index += 1

        j+=1
        if j>=len(board2) and i<=len(board2)-2:
            j=0
            i+=1
    return result



def emptyTilesCount(board):
    flat_list = [item for sublist in board for item in sublist]
    return flat_list.count(' ')


def difference_tiles(flat_list, player):
    sum = 0
    sum1 = 0
    weights = {'a' : 1, 'b' : 2, 'c' : 3, 'd' : 4, 'e' : 5, 'f': 6, 'g':7, 'h':8, 'i':9, 'j':10, 'k':11,
               'A' : 1, 'B' : 2, 'C' : 3, 'D' : 4, 'E' : 5, 'F': 6, 'G':7, 'H':8, 'I':9, 'J':10, 'K':11, ' ':0}
    for i in range(0, len(flat_list)):
        if ord(flat_list[i]) >= 65 and ord(flat_list[i]) <= 90 and flat_list[i] != ' ':
            sum += (weights[flat_list[i]]*ord(flat_list[i]))
        else:
            sum1 += (weights[flat_list[i]]*ord(flat_list[i]))
    if player == 0:
        return (sum - sum1)
    else:
        return (sum1 - sum)

def max_tiles(flat_list, player):
    if player == 0:
        max = 'A'
        for i in range(0, len(flat_list)):
            if ord(flat_list[i]) >= 65 and ord(flat_list[i]) <= 90 and flat_list[i] != ' ':
                if ord(max) < ord(flat_list[i]):
                    max = flat_list[i]
    else:
        max = 'a'
        for i in range(0, len(flat_list)):
            if ord(flat_list[i]) >= 97 and ord(flat_list[i]) <= 107 and flat_list[i] != ' ':
                if ord(max) < ord(flat_list[i]):
                    max = flat_list[i]

    return ord(max)


def heuristic(board, player):
    flat_list = [item for sublist in board for item in sublist]

    return difference_tiles(flat_list, player) + flat_list.count(' ')



def expectiminimax(fringe):
    fringeFinal = deepcopy(fringe)
    parentNodes = []
    heuristicValueList = []
    maxValueForInitialNode = -999999
    tryVal = 3
    chanceFlag = -1

    count = 0
    numberOfEmptySpaces=0
    for i in range(len(fringeFinal)-1, -1, -1):
        parentValue = fringeFinal[i][6]
        minMaxValue = fringeFinal[i][1]
        heuristicValue = fringeFinal[i][3]
        currentBoard = fringeFinal[i][0]
        currentIndexValue = fringeFinal[i][5]
        if parentValue >= 1:
            if currentIndexValue in parentNodes and minMaxValue not in [0,1]:
                numberOfEmptySpaces = emptyTilesCount(currentBoard)
                try:
                    heuristicValue = (1/numberOfEmptySpaces * heuristicValueList[parentNodes.index(currentIndexValue)])
                except:
                    heuristicValue = -1

                indexValue = parentNodes.index(currentIndexValue)

                heuristicValueList[indexValue] = heuristicValue
                if parentValue not in parentNodes:
                    count_empty_tiles = emptyTilesCount(fringeFinal[i][4])-1
                    parentNodes[parentNodes.index(currentIndexValue)] = parentValue
                else:
                    count_empty_tiles -= 1
                    indexValue = parentNodes.index(parentValue)
                    heuristicValueList[indexValue] += heuristicValue
                    if count_empty_tiles == 0:
                        try:
                            heuristicValueList[indexValue] = heuristicValueList[indexValue]/emptyTilesCount(fringeFinal[i][4])
                        except:
                            heuristicValueList[indexValue] = -1

                

            elif parentValue not in parentNodes:
                if minMaxValue == 1:
                    chanceFlag=0
                elif minMaxValue == 0:
                    chanceFlag=1
                parentNodes.append(parentValue)
                heuristicValueList.append(heuristicValue)
            else:
                indexValue = parentNodes.index(parentValue)
                if (minMaxValue == 0 and heuristicValueList[indexValue] < heuristicValue):
                    chanceFlag = 1
                    heuristicValueList[indexValue] = heuristicValue
                elif (minMaxValue == 1 and heuristicValueList[indexValue] > heuristicValue):
                    chanceFlag = 0
                    heuristicValueList[indexValue] = heuristicValue
        elif parentValue==0 and currentIndexValue!=0:
            indexValue = parentNodes.index(currentIndexValue)

            if heuristicValueList[indexValue] >= maxValueForInitialNode:
                maxValueForInitialNode = heuristicValueList[indexValue]

                count  = tryVal
            tryVal = tryVal-1
            if tryVal==0:
                return count
    return count


def next_move(game: Game_IJK)-> None:

    '''board: list of list of strings -> current state of the game
       current_player: int -> player who will make the next move either ('+') or -'-')
       deterministic: bool -> either True or False, indicating whether the game is deterministic or not
    '''

    board = game.getGame()
    player = game.getCurrentPlayer()
    deterministic = game.getDeterministic()


    global current_index

    if deterministic:
        fringe = [board]
        count = 1

        while len(fringe) >= 0 and count<=85:

            board = fringe.pop(0)
            fringe+=find_successors(board, player, deterministic)
            count+=1

        list_mini_max = []
        for i in fringe:
            list_mini_max.append(heuristic(i, player))

        temp,temp1 = minimax(0, 0, True, list_mini_max, -1000, 1000)

        temp_dict = {0 : 'L', 1: 'R', 2: 'U', 3: 'D'}
        yield temp_dict[temp1]

    else:
        # Tuple structure : Board, max=1/min=0/chance=-1, depth, Heuristic, parent, current node index, parent node index
        fringe = [(board, 1, 1, 0, "root", 0, 0)]
        fringe1 = [(board, 1, 1, 0, "root",0, 0)]
        tree_depth = 4
        empty_tiles = 1
        board = fringe.pop(0)
        succ = []
        current_index = 1
        succ += find_successors1(board[0], player, board[2], deterministic, tree_depth, board[5])

        for i in succ:
            fringe1.append(i)
            fringe.append(i)
            current_index += 1
        
        chance_flag = 0
        while len(fringe) > 0:
            fringe_temp = []
            board = fringe.pop(0)

            if board[2] < tree_depth:
                if board[1] in [1,0]:
                    fringe_temp = [chance_nodes for chance_nodes in find_chance(board[0], board[1], board[2], board[5])]
                    fringe += fringe_temp
                    fringe1 += fringe_temp
                    chance_flag = board[1]

                else:
                    fringe_temp += find_successors1(board[0], chance_flag, board[2], deterministic, tree_depth, board[5])
                    fringe += fringe_temp
                    fringe1 += fringe_temp
            else:
                break

        count =  expectiminimax(fringe1)
        dict = {0 : 'L', 1: 'R', 2: 'U', 3: 'D'}

        yield dict[count]


