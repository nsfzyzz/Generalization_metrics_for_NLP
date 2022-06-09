import argparse

def main(args):

    with open(args.file_name, 'r') as f:
        all_commands = f.readlines()
    
    file_length = len(all_commands)
    each_file_length = file_length//args.num_splits+1

    row = 0
    for file_ind in range(args.num_splits):
        commands_split_file = []
        for command_ind in range(each_file_length):
            commands_split_file.append(all_commands[row])
            row += 1
            if row>=file_length:
                break
        with open(args.file_name[:-3]+"_"+str(file_ind)+".sh", "w") as fwrite:
            for line in commands_split_file:
                fwrite.write(line)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", type=str, help="file to parse")
    parser.add_argument("--num-splits", type=int, default=3, help="number of files to split")
    args = parser.parse_args()
    
    print("Arguments for the experiment.")
    for arg in vars(args):
        print(arg, getattr(args, arg))
    
    main(args)
