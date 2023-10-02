import argparse

if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument("--name", "-n", default="shubh",help=str, type=str)
    args.add_argument("--age", "-a", default=25.0, help=str, type=float)
    parse_args=args.parse_args()

    print(parse_args.name,parse_args.age)
    
