/*
   american fuzzy lop - a trivial program to test the build
   --------------------------------------------------------

   Written and maintained by Michal Zalewski <lcamtuf@google.com>

   Copyright 2014 Google Inc. All rights reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at:

     http://www.apache.org/licenses/LICENSE-2.0

 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int foo(); 
int bar(); 


int main(int argc, char** argv) {

  char buf[8];

  if (read(0, buf, 8) < 1) {
    printf("Hum?\n");
    exit(1);
  }

  int c = foo(); 
  if (buf[0] == '0')
    printf("Looks like a zero to me!\n");
  else
    printf("A non-zero value? How quaint!\n");
  int x = bar();
  int y = foo();  
  exit(0);
  printf("Let's c=%d", x+y); 
}

int foo() {

  int a = 6;
  int b; 
  while (a >= 0) {
      b = a * (a - 1);
      a -= 1;
      printf("%d\n", b);
  }
  b += bar();
  printf("Foo...\n");
  return b; 
}

int bar() {
  printf("... Bar\n"); 
  int a = 7;
  int b; 
  while (a >= 0) {
      b = a * (a + 1);
      a -= 1;
      printf("%d\n", b);
  }
  return b;
}
