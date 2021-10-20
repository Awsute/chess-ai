from math import pi
import re


from graphics import *
from os import system
import time
import chess
import chess.pgn
from neuralnet import *
#a1 is dark

#[
# board values:
# 0 = empty space
# 1 = white king
# 2 = white queen
# 3 = white bishop
# 4 = white knight
# 5 = white rook
# 6 = white pawn
#
# -1 = black king
# -2 = black queen
# -3 = black bishop
# -4 = black knight
# -5 = black rook
# -6 = black pawn
#
#
#]
#light squares are 1
#dark squares are 0




def draw_board(win, square_size, 
                board, board_col,
                off, 
                w_pieces, b_pieces, 
                w_col, b_col):
    s = [square_size[0], square_size[1]]
    for row in range(0, len(board)):
        y = s[1]*row+off[1]
        for column in range(0, len(board[row])):

            square = board[row][column]

            x = s[0]*column+off[0]


            piece = square[1]


            rect = Rectangle(Point(x, y), Point(x+s[0], y+s[1]))

            if square[2]:
                rect.setFill(color_rgb(150, 150, 0))
            elif square[0] == 0: #dark square
                rect.setFill(board_col[0])
            elif square[0] == 1: #light square
                rect.setFill(board_col[1])
            
            rect.setWidth(6)
            rect.draw(win)

            text = Text(Point(x + s[0]/2, y + s[1]/2), "")
            text.setSize(18)
            o_col = ""
            txt_out = 3
            
            if piece != 0:
                if piece > 0:
                    text.setText(w_pieces[piece-1])
                    text.setTextColor(w_col[0])
                    o_col = w_col[1]
                else:
                    text.setText(b_pieces[-piece-1])
                    text.setTextColor(b_col[0])
                    o_col = b_col[1]
                

                for i in range(0, 4):
                    for o in range(0, 4):
                        x_t = (i-2)/4*txt_out
                        y_t = (o-2)/4*txt_out
                        t = Text(Point(x + s[0]/2, y + s[1]/2), "")
                        t.move(x_t, y_t)
                        t.setStyle("bold")
                        t.setTextColor(o_col)
                        t.draw(win)
                        
            
            text.draw(win)


white = ["K", "Q", "B", "N", "R", "P"]
black = ["k", "q", "b", "n", "r", "p"]
def fen_to_brd(fen, brd):

    inf = fen.split()
    info = ""
    for i in range(1, len(inf)):
        info += inf[i]
    
    s = inf[0].split("/")
    for r in range(0, 8):
        row = []
        ps = s[r]
        column = 0
        for p in ps:
            
            pc = 0
            if p.upper() == p:
                for w in range(0, len(white)):
                    if white[w] == p:
                        pc = w+1
                        break
            else:
                for b in range(0, len(black)):
                    if black[b] == p:
                        pc = -(b+1)
                        break
            if p.isnumeric():
                for o in range(0, int(p)):
                    brd[7-r][column][1] = 0
                    column += 1
            else:
                brd[7-r][column][1] = pc
                column += 1
        brd.append(row)
    return brd, info

abcs = "abcdefghijklmnopqrstuvwxyz"
def move_to_letter(pos_x, pos_y, tar_x, tar_y):
    return abcs[pos_x]+str(pos_y+1)+abcs[tar_x]+str(tar_y+1)

def move_to_num(mv):
    return [abcs.find(mv[0]), int(mv[1])-1, abcs.find(mv[2]), int(mv[3])-1]


def new_board():
    b = []
    for i in range(0, 8):
        r = []
        for o in range(0, 8):
            r.append([(i+o)%2, 0, False])
        b.append(r)
    return b

def get_piece(piece):
    if piece == "P":
        return 6
    elif piece == "R":
        return 5
    elif piece == "N":
        return 4
    elif piece == "B":
        return 3
    elif piece == "Q":
        return 2
    elif piece == "K":
        return 1


    elif piece == "p":
        return -6
    elif piece == "r":
        return -5
    elif piece == "n":
        return -4
    elif piece == "b":
        return -3
    elif piece == "q":
        return -2
    elif piece == "k":
        return -1
    
    else:
        return 0

def separate_fen(fen):
    l = fen.split()
    rs = []
    brd = l[0].split("/")
    #board
    for i in range(0, len(brd)):
        for j in range(0, len(brd[i])):
            if brd[i][j].isnumeric():
                for o in range(0, int(brd[i][j])):
                    rs.append(1)
            else:
                rs.append(get_piece(brd[i][j])/6)
    
    #turn
    if l[1]== "w":
        rs.append(1)
    else:
        rs.append(0)
    
    #castling
    if l[2].find("KQ") == -1:
        rs.append(0)
    else:
        rs.append(1)
    
    if l[2].find("kq") == -1:
        rs.append(0)
    else:
        rs.append(1)
    
    #turn counter
    if l[4].isnumeric():
        rs.append(int(l[4]))
    else:
        rs.append(-1)
    
    if l[5].isnumeric():
        rs.append(int(l[5]))
    else:
        rs.append(-1)

    return rs

        
def pos_to_num(pos_x, pos_y):
    return pos_x+8*pos_y


brd = chess.Board()
win = GraphWin("test", 1600, 800, False)
win.setCoords(0, 0, win.width, win.height)
win.plot(20, 20)
win.setBackground(color_rgb(50, 15, 50))


net1 = Network(69, [], Activator(lambda x: safe_sigmoid(x), lambda x: (1-safe_sigmoid(x))*safe_sigmoid(x)))
#input is fen separated into 69 parts
net1.hidden = net1.random_net(1)
#net1 = net1.import_from_file("current_ai.json")
#out1 is index of possible move


#tournament bracket for gecnetic algorithm type thing
def setup_bracket(base_net, size):
    b = []
    r = []
    for i in range(2**size):
        b1 = Network(len(base_net.inputs), base_net.hidden.copy(), base_net.activator)
        b1.backprop([0], [[random()]])

        b2 = Network(len(base_net.inputs), base_net.hidden.copy(), base_net.activator)
        b2.backprop([0], [[random()]])

        b.append([b1, b2])
        r.append([0, 0])
    return b, r

def setup_bracket_rand(base_net, size):
    b = []
    r = []
    for i in range(2**size):
        b1 = Network(len(base_net.inputs), base_net.hidden.copy(), base_net.activator)
        b1.backprop([0], [[random()]])

        b2 = Network(69, [], base_net.activator)
        b2.hidden = b2.random_net(1)
        #if randint(0, 1) == 0:
        b.append([b1, b2])
        #else:
        #    b.append([b2, b1])
        r.append([0, 0])
    return b, r

def bracket_next_layer(bracket, records):
    b = []
    r = []
    for i in range(int(len(bracket)/2)):
        winner1 = bracket[i][0]
        if records[i][1] > records[i][0]:
            winner1 = bracket[i][1]
        
        winner2 = bracket[i+1][0]
        if records[i+1][1] > records[i+1][0]:
            winner2 = bracket[i+1][1]
        b.append([winner1, winner2])
        r.append([0, 0])
    return b, r
        
        


board_col = [color_rgb(25, 75, 25), color_rgb(200, 200, 200)] #[dark, light]
w_piece_col = [color_rgb(255, 255, 255), color_rgb(0, 0, 0)]
b_piece_col = [color_rgb(0, 0, 0), color_rgb(255, 255, 255)]

square_size = [75, 75]
board_offest = [100, 100]

in_square = lambda point : [int((point.getX()-board_offest[0])/square_size[0]), int((point.getY()-board_offest[1])/square_size[1])]

turn = 0
#light is turn 0
#dark is turn 1
bracket_layers = 3
selected = []
board = new_board()
bracket, records = setup_bracket_rand(net1, bracket_layers)

players = [net1, net1]

training = True
#WARNING FOR ANYONE TRYING THIS OUT: ai will go through a "really short game" phase (games under 20 moves) starting at generation 1200-1300 and ending in about 100-200 generations
#starting generation and length of phase depend on dimensions of neural net
#i have added some measures to combat this but they do not eliminate it completely
bracket_num = 0
game_counter = 0.0
bracket_layer_count = 0
while win.checkKey() != "Escape":

    fen = brd.fen()
    board, info = fen_to_brd(fen, board)
    #board_dim = draw_board(win, square_size, board, board_col, board_offest, white, black, w_piece_col, b_piece_col)
    
    if brd.is_game_over():
        if training:
            if game_counter == len(bracket):
                print(game_counter)
                game_counter = 0
                
                if bracket_layer_count == bracket_layers:
                    winner = bracket[int(game_counter)-1][0]
                    if records[int(game_counter)-1][1] > records[int(game_counter)-1][0]:
                        winner = bracket[int(game_counter)-1][1]
                    bracket, records = setup_bracket_rand(winner, bracket_layers)
                    winner.output_to_file("current_ai.json")
                    
                    if bracket_num%100 == 0:
                        game = chess.pgn.Game.from_board(brd)
                        # Undo all moves.
                        switchyard = []
                        while brd.move_stack:
                            switchyard.append(brd.pop())

                        game.setup(brd)
                        node = game

                        # Replay all moves.
                        while switchyard:
                            move = switchyard.pop()
                            node = node.add_variation(move)
                            brd.push(move)
                        game.headers["Result"] = brd.result()
                        with open("games/game" + str(bracket_num) + "-" + str(bracket_layer_count) + "-" + str(game_counter) + "-64-8-RL" + ".pgn", "x") as f:
                            f.write(game.__str__())
                    
                    bracket_layer_count = 0
                    bracket_num += 1
                    print(bracket_num)
                else:
                    bracket_layer_count += 1
                    bracket, records = bracket_next_layer(bracket, records)
            else:
                result = brd.result()
                if brd.is_checkmate():
                    if float(game_counter).is_integer():
                        records[int(game_counter)][0] += int(result[0])
                        records[int(game_counter)][1] += int(result[len(result)-1])
                    else:
                        records[int(game_counter)][1] += int(result[0])
                        records[int(game_counter)][0] += int(result[len(result)-1])

                
                print(records)
                if float(game_counter).is_integer():
                    players[0] = bracket[int(game_counter)][0]
                    players[1] = bracket[int(game_counter)][1]
                else:
                    players[0] = bracket[int(game_counter)][1]
                    players[1] = bracket[int(game_counter)][0]
                game_counter += 0.5
            

        brd.reset()
        turn  = 0
    else:
        
        if players[turn] == True:
            m_pos = win.checkMouse()
            if m_pos:

                
                    [x, y] = in_square(m_pos)
                    if y >= 0 and y < len(board):
                        if x >= 0 and x < len(board[y]):
                            if len(selected) == 0:
                                board[y][x][2] = True
                                selected = [x, y]
                                
                            elif x != selected[0] or y != selected[1]:
                                board[selected[1]][selected[0]][2] = False
                                move = move_to_letter(selected[0], selected[1], x, y)
                                mv = chess.Move.from_uci(move)
                                if brd.is_legal(mv):
                                    brd.push(mv)
                                    selected = []
                                    turn += 1
                                    turn = turn%2
                                else: 
                                    selected = [x, y]
                                    board[y][x][2] = True
                                
                            elif x == selected[0] and y == selected[1]:
                                board[selected[1]][selected[0]][2] = False
                                selected = []
        else:
            net = players[turn]
            lgl_moves = brd.generate_legal_moves()
            j = []
            for m in lgl_moves:
                n = m.uci()
                j.append(n)
            tries = 0
            
            g = net.output(separate_fen(fen))

            m = g[len(g)-1][0]
            move = j[int(m*len(j))]
            mv = chess.Move.from_uci(move)
            brd.push(mv)
            turn += 1
            turn = turn%2

                        
                
    time.sleep(0)