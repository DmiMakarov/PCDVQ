from .container import PCDVQCodebook

def main():
    codebook = PCDVQCodebook()
    codebook.build()
    codebook.save('codebook.pt')

if __name__ == "__main__":
    main()
