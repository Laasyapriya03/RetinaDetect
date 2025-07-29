// This is a normal comment
#include <stdio.h> // Include directive
#define MAX_BUFFER_SIZE 1024 // Define directive

#ifdef DEBUG_MODE // Ifdef directive
void debug_log(const char* msg) {
    // Debug message
    printf("[DEBUG] %s\n", msg);
}
#else // Else directive
void release_log(const char* msg) {
    // Release message
    printf("[RELEASE] %s\n", msg);
}
#endif // Endif directive

#if VERSION == 1 // If directive
const char* app_version = "v1.0";
#elif VERSION == 2 // Elif directive
const char* app_version = "v2.0";
#else
const char* app_version = "unknown";
#endif

#pragma once // Pragma directive
#error This is a custom error directive // Error directive

int main() {
    printf("Hello, World!\n");
    return 0;
}