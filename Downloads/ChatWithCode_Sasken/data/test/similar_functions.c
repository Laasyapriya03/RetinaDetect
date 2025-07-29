#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// Reverses a string in-place
void reverse_string(char* str) {
    int len = strlen(str);
    for (int i = 0; i < len / 2; i++) {
        char tmp = str[i];
        str[i] = str[len - i - 1];
        str[len - i - 1] = tmp;
    }
}

// Reverses an array of integers
void reverse_array(int arr[], int size) {
    for (int i = 0; i < size / 2; i++) {
        int temp = arr[i];
        arr[i] = arr[size - i - 1];
        arr[size - i - 1] = temp;
    }
}

// Reverses each word in a string separately
void reverse_words(char* sentence) {
    char* start = sentence;
    while (*start) {
        char* end = start;
        while (*end && *end != ' ') end++;
        int len = end - start;
        for (int i = 0; i < len / 2; i++) {
            char tmp = start[i];
            start[i] = start[len - i - 1];
            start[len - i - 1] = tmp;
        }
        if (*end) start = end + 1;
        else break;
    }
}

// Sorts an array (bubble sort)
void sort_array(int arr[], int size) {
    for (int i = 0; i < size-1; i++) {
        for (int j = 0; j < size-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                int temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }
}

// Shuffles an array (Fisher–Yates)
void shuffle_array(int arr[], int size) {
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

// Copies one array to another
void copy_array(int dest[], int src[], int size) {
    for (int i = 0; i < size; i++) {
        dest[i] = src[i];
    }
}

// Prints an array
void print_array(int arr[], int size) {
    printf("Array: ");
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}
