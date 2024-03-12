#include <stdio.h>

int square(int i) {
  return i * i;
}

int main() {
  int x;
  int y;
  x = 2;
  y = square(x);
  printf("%d", y);
  return 0;
}
