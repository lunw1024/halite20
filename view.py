import tkinter as tk
from tkinter.filedialog import askopenfile
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
import json

def from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
    """
    return "#%02x%02x%02x" % rgb   

#Board
boardInfo = [[{"x":x,"y":y,"background":(0,0,0),"unit":None,"unitColor":None,"extra":[]}for y in range(21)] for x in range(21)]
game = None
turn = 0
colorMap = [(255,255,0),(255,0,0),(0,255,0),(128,0,128),(225,225,0),(225,0,0),(0,225,0),(108,0,108)]

def unpack(n,size):
    return n // size, n % size
def pack(x,y,size):
    return x * size + y

class Point():

    def __init__(self,info,*args, **kwargs):
        global board
        self.frame = tk.Frame(master=board,width=30,height=30,relief="raised",border=1,background=from_rgb(info['background']))
        self.frame.grid_propagate(0)
        self.frame.bind("<Button-1>",self.popup)
        self.frame.grid(row=info['y'],column=info['x'])
        self.label = tk.Label(master=self.frame,bg=from_rgb(info['background']))
        self.label.bind("<Button-1>",self.popup)
        self.label.grid()
        self.info = info
        # Unit
        #1 - up, 2 - right, 3 - down, 4 - left
        if info['unit'] == 1:
            self.label.configure(text="▲")
        if info['unit'] == 2:
            self.label.configure(text="▶")
        if info['unit'] == 3:
            self.label.configure(text="▼")
        if info['unit'] == 4:
            self.label.configure(text="◀")  
        if info['unitColor'] != None:
            self.label.configure(fg=from_rgb(info['unitColor']))
    def update(self,info):
        self.info = info
        self.frame.configure(background=from_rgb(info['background']))
        self.label.configure(background=from_rgb(info['background']))
        if info['unit'] == 1:
            self.label.configure(text="▲")
        elif info['unit'] == 2:
            self.label.configure(text="▶")
        elif info['unit'] == 3:
            self.label.configure(text="▼")
        elif info['unit'] == 4:
            self.label.configure(text="◀")  
        else:
            self.label.configure(text="")  
        if info['unitColor'] != None:
            self.label.configure(fg=from_rgb(info['unitColor']))

    def popup(self,event):
        popup = tk.Tk()
        popup.geometry("400x500")
        popup.title(str(self.info["x"])+","+str(self.info["y"]))
        label = tk.Label(popup, text=self.info["extra"])
        label.pack(side="top", fill="x", pady=10)
        B1 = tk.Button(popup, text="Close", command = popup.destroy)
        B1.pack()
        popup.mainloop()


pointBoard = []
def refresh():
    global pointBoard
    if len(board.winfo_children()) != len(boardInfo) ** 2:
        for widget in board.winfo_children():
            widget.destroy()
        for x in boardInfo:
            pointBoard.append([])
            for y in x:
                pointBoard[-1].append(Point(y))
    else:
        for x in boardInfo:
            for y in x:
                target = pointBoard[y['x']][y['y']]
                target.update(y)
            
def update():
    global boardInfo,turn

    print("Updating")

    if game == None:
        print("No game found!")
        return
    
    gameLength = len(game['steps'])
    if turn >= gameLength:
        turn = gameLength-1
    if turn < 0:
        turn = 0
    
    turnLabel.configure(text=str(turn))
    turnInfo = game['steps'][turn]
    n = game['configuration']['size']
    me = 0
    boardInfo = [[{"x":x,"y":y,"background":(0,0,0),"unit":None,"unitColor":None,"extra":[]}for y in range(n)] for x in range(n)]
    haliteMap = turnInfo[me]['observation']['halite']

    prev = {}
    # Units
    shipMap=[[None for y in range(n)] for x in range(n)]
    shipyardMap = [[None for y in range(n)] for x in range(n)]
    for team in range(len(turnInfo[me]['observation']['players'])):
        teams[team].configure(text= turnInfo[me]['observation']['players'][team][0]) 
        shipRaw = turnInfo[me]['observation']['players'][team][2]
        for ship in shipRaw:
            s = shipRaw[ship]
            x,y = unpack(s[0],n)
            sHalite = s[1]
            shipMap[x][y] = (team,sHalite,ship)
            prev[ship] = 1
            if turn > 0:
                try:
                    prevShip = game['steps'][turn-1][me]['observation']['players'][team][2][ship]
                    xx,yy = unpack(prevShip[0],n)
                    if xx < x:
                        prev[ship] = 2
                    elif xx > x:
                        prev[ship] = 4
                    elif yy > y:
                        prev[ship] = 3    
                except:
                    pass
        shipyardRaw = turnInfo[me]['observation']['players'][team][1]
        for shipyard in shipyardRaw:
            s = shipyardRaw[shipyard]
            x,y = unpack(s,n)
            shipyardMap[x][y] = team

    #Update board
    
    for x,col in enumerate(boardInfo):
        for y,point in enumerate(col):
            color = int(haliteMap[pack(x,y,n)] / game['configuration']['maxCellHalite'] * 255)
            point['background'] = (color,color,color)

            unit = "None"
            unitHalite = "None"

            if shipMap[x][y] != None:
                point['unit'] = prev[shipMap[x][y][2]]
                point['unitColor'] = colorMap[shipMap[x][y][0]]
                unit = shipMap[x][y][0]
                unitHalite = shipMap[x][y][1]
            if shipyardMap[x][y] != None:
                point['background'] = colorMap[shipyardMap[x][y]+4]
            point['extra'] = "\n".join([
                "Unit: " + str(unit),
                "Unit Halite: " + str(unitHalite)
            ])

    refresh()


def right(event):
    global turn
    turn += 1
    update()

def left(event):
    global turn
    turn -= 1
    update()

#GUI
root = tk.Tk()
root.title("Syncbot Viewer")
root.resizable(0, 0)
root.geometry("1040x640")
main = tk.Frame(master=root,width=1040,height=640)
main.pack(expand=True, fill='both')
main.pack_propagate(0)

board = tk.Frame(master=main,width=640,height=640,relief="raised",border=5,background="gray12")
board.grid(row=0,column=0)

display = tk.Frame(master=main,width=400,height=640,relief="raised",border=5,background="gray12")
display.grid(row=0,column=1)

display.grid_propagate(0)
display.grid_columnconfigure(0, minsize=50) 
display.grid_columnconfigure(1, minsize=50) 
display.grid_columnconfigure(2, minsize=50) 
display.grid_columnconfigure(3, minsize=50) 
board.grid_propagate(0)

tk.Label(display,text="DATA",fg="gray35",bg="gray12",font=("Courier", 44)).grid(row=0,column=0)

def inputNewGame():
    global game
    print("Input")
    f = askopenfile("r",filetypes=[("JSON",'*.json')])
    try:
        game = json.loads(f.read())
        print("Success?")
    except:
        print("Cannot load file")
    update()
    refresh()

tk.Button(master=display,text="Game JSON",command=inputNewGame).grid(row=1,column=0)

turnLabel = tk.Label(display,text="TURN",fg="gray35",bg="gray12",font=("Courier", 44))
turnLabel.grid(row=2,column=0)
tk.Label(display,text="Halite",fg="gray35",bg="gray12",font=("Courier", 44)).grid(row=3,column=0)

teams = [tk.Label(display,text="0",fg=from_rgb(colorMap[i]),bg="gray12",font=("Courier", 20)) for i in range(4)]
for i, team in enumerate(teams):
    team.grid(row=4,column=i)

refresh()

root.bind('<Right>', right)
root.bind('<Left>', left)
root.mainloop()



