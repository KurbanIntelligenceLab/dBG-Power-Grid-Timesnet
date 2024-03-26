#include <stdlib.h>
#include <stdio.h>

int min(int a, int b, int c) {
    int m = a;
    if (b < m) m = b;
    if (c < m) m = c;
    return m;
}

int levenshtein_distance(int *array1, int len1, int *array2, int len2) {
    int i, j;
    int **d = (int **)malloc((len1 + 1) * sizeof(int *));
    for (i = 0; i <= len1; i++) {
        d[i] = (int *)malloc((len2 + 1) * sizeof(int));
    }

    for (i = 0; i <= len1; i++) d[i][0] = i;
    for (j = 0; j <= len2; j++) d[0][j] = j;

    for (i = 1; i <= len1; i++) {
        for (j = 1; j <= len2; j++) {
            int cost = abs(array1[i - 1] - array2[j - 1]);
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost);
        }
    }

    int result = d[len1][len2];
    for (i = 0; i <= len1; i++) {
        free(d[i]);
    }
    free(d);

    return result;
}
