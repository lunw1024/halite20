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

    for file in files:
        if file == folder + "/agent.py":
            continue
        f = open(file,"r")
        output.write(f.read())
        output.write("\n")
        f.close()
    f = open(folder + "/agent.py","r")
    output.write(f.read())
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
