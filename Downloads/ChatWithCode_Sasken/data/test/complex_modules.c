#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// Initializes the system
void init_system() {
    printf("System initialized.\n");
}

// Adds two numbers
int add(int a, int b) {
    return a + b;
}

// Multiplies two numbers
int multiply(int a, int b) {
    return a * b;
}

// Recursive factorial function
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

// Checks if a number is prime
int is_prime(int n) {
    if (n <= 1) return 0;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) return 0;
    }
    return 1;
}

// Prints all prime numbers up to n
void print_primes(int n) {
    printf("Primes up to %d:\n", n);
    for (int i = 2; i <= n; i++) {
        if (is_prime(i)) {
            printf("%d ", i);
        }
    }
    printf("\n");
}

// Copies a string
void copy_string(char* dest, const char* src) {
    while (*src) {
        *dest++ = *src++;
    }
    *dest = '\0';
}

// Reverses a string in place
void reverse_string(char* str) {
    int len = strlen(str);
    for (int i = 0; i < len / 2; i++) {
        char tmp = str[i];
        str[i] = str[len - i - 1];
        str[len - i - 1] = tmp;
    }
}

// Simulates a task runner
void run_tasks() {
    for (int i = 1; i <= 5; i++) {
        printf("Running task %d...\n", i);
    }
}

// Handles user login
int login(const char* username, const char* password) {
    if (strcmp(username, "admin") == 0 && strcmp(password, "pass") == 0) {
        return 1; // success
    }
    return 0; // failure
}

// Some inline code to test chunk separation
int x = 5;
char global_message[100] = "Hello from global scope!";
