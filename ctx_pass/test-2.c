#include  <stdio.h>
#include <stdlib.h> 

int func_a();
int func_b();
int func_c(int val);
int func_d(int val);

int main(int argc, char** argv) {
    int arg_1 = atoi(argv[1]);
    char comment1[] = "Initilize a comment string";
    char comment2[] = "Set an intermediate value";
    int intermediate_val_1;
    if (arg_1 >= 1000)
    {
        char comment3[] = "If the argument is larger than 1000 call func_a()";
        intermediate_val_1 = func_a(arg_1);
    }
    else
    {
        char comment4[] = "If the argument is less than 1000 call func_b()";
        intermediate_val_1 = func_b(arg_1);
    }

    char comment5[] = "Call func_c()";
    int intermediate_val_2 = func_c(intermediate_val_1);
    char comment6[] = "Call func_d()";
    int intermediate_val_3 = func_d(intermediate_val_2);
    char comment7[] = "The end";
    return 0; 
}

int func_a() {
    return 10000; 
}

int func_b() {
    return -10000;
}

int func_c(int val) {
    return val * 5;
}

int func_d(int val) {
    return val / 10; 
}
