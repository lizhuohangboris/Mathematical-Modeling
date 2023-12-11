class ArrayProcessor:
    def __init__(self, arr):
        self.array = arr

    def construct(self):
        print("Array elements:")
        for item in self.array:
            print(item)
        print("Construction complete.")

    def destruct(self, index, condition):
        if 0 <= index < len(self.array):
            item = self.array[index]
            if condition(item):
                print(f"Item at index {index} satisfies the condition.")
            else:
                print(f"Item at index {index} does not satisfy the condition.")
        else:
            print("Invalid index.")

# 示例用法
def main():
    my_array = [10, 25, 30, 45, 50]
    processor = ArrayProcessor(my_array)
    
    processor.construct()
    
    index_to_check = 2
    condition_to_check = lambda x: x > 20
    processor.destruct(index_to_check, condition_to_check)

if __name__ == "__main__":
    main()
