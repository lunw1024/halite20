import glob, sys, getopt

def build(folder):
    print("Building", folder)
    files = glob.glob(folder+"/*.py")

    if len(files) == 0:
        print("Empty! No files in folder ", folder)
        return
    
    if not folder + "/agent.py" in files:
        print("Error! Required file agent.py")
        return 

    output = open(folder+".py","w").close()
    output = open(folder+".py","a")

    f = open(folder + "/dependency.py","r")
    for line in f:
        output.write(line)
    output.write("\n")
    f.close()

    for file in files:
        if file == folder + "/agent.py" or file == folder+"/dependency.py":
            continue
        f = open(file,"r")
        for line in f:
            #Ignore all import statements
            if line.startswith('from') or line.startswith ('import'):
                continue
            output.write(line)
        output.write("\n")
        f.close()
    #Final agent.py
    f = open(folder + "/agent.py","r")
    for line in f:
        #Ignore all import statements
        if line.startswith('from') or line.startswith ('import'):
            continue
        output.write(line)
    f.close()

    output.close()

if __name__ == "__main__":
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"h",["ifolder="])
        for opt, arg in opts:
            if opt == '-h':
                print ('build.py <inputfolder>')
                sys.exit()
        if len(args) != 1:
            raise getopt.GetoptError("Error.")
        inputFolder = args[0]
        build(inputFolder)
    except getopt.GetoptError:
        print ('Error: build.py <inputfolder>')
        sys.exit(2)
