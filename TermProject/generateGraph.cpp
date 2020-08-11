//
// Created by Pedzinski, Dale on 8/9/20.
//
#include "jsmn.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <limits.h>       //For PATH_MAX

//// Structure to represent a vertex and its distance
//struct distNode {
//    int64_t node;
//    int64_t dist;
//
//    bool operator<(const distNode &rhs) const {
//        return dist > rhs.dist || (dist == rhs.dist && node > rhs.node);;
//    }
//};
//
//// Structure to represent an edge
//struct edge {
//    int64_t from;
//    int64_t to;
//    int64_t weight;
//
//    bool operator<(const edge &rhs) const {
//        return weight > rhs.weight || (weight == rhs.weight && to > rhs.to);
//    }
//};
//
//// Structure to represent a edge source & destination
//struct fromTo {
//    int64_t from;
//    int64_t to;
//
//    bool operator<(const fromTo &rhs) const {
//        return to < rhs.to || (to == rhs.to && from < rhs.from);
//    }
//};

const char *get_filename_ext(const char *filename) {
    const char *dot = strrchr(filename, '.');
    if (!dot || dot == filename) return "";
    return dot + 1;
}

int main() {
    struct dirent *de;  // Pointer for directory entry

    // opendir() returns a pointer of DIR type.
    DIR *dr = opendir(".");

    if (dr == NULL)  // opendir returns NULL if couldn't open directory
    {
        printf("Could not open current directory\n");
        return 0;
    }

    // Refer http://pubs.opengroup.org/onlinepubs/7990989775/xsh/readdir.html
    // for readdir()

    while ((de = readdir(dr)) != NULL) {
        char filePath[PATH_MAX + 1];
        if (strncmp(get_filename_ext(de->d_name), "csv", 2) == 0) {
            realpath(de->d_name, filePath);
            printf("%s\n", filePath);
            char const *const fileName = filePath;
            FILE *file = fopen(fileName, "r"); /* should check the result */
            printf("%s\n", "file opened");
            char line[4000];

            int state = 0;
            int subState = 0;
            while (fgets(line, sizeof(line), file)) {
                char *token;
                char *rest = line;
                int *source;
                int *target;
                int *value;
//                printf("%s\n", "Next line:::");
                while ((token = strtok_r(rest, "|\n", &rest))) {
//                    printf("%s\n", token);
                    if (strncmp(token, "Nodes", 2) == 0) {
                        printf("Section is %s\n", token);
                        state = 1;
                        subState=0;
                        break;
                    }
                    if (strncmp(token, "Edges", 2) == 0) {
                        printf("Section is %s\n", token);
                        state = 2;
                        subState=0;
                        break;
                    }
                    if (strncmp(token, "Operator", 2) == 0) {
                        printf("Section is %s\n", token);
                        state = 3;
                        subState=0;
                        break;
                    }
                    if (state == 1 && subState==0 ) {
                        printf("Node is %s\n", token);
                        break;
                    }

                    if (state == 2 && subState == 0) {
                        //                            Source
                        printf("Source is %s\n", token);
                        source=(int *)token;
                        subState = 1;
                    }else if (state == 2 && subState == 1) {
                        //                            Target
                        printf("Target is %s\n", token);
                        target =(int *)token;
                        subState = 2;
                    }else if (state == 2 && subState == 2) {
                        //                            Value
                        printf("Value is %s\n", token);
                        value=(int *)token;
                        subState = 0;
                        break;
                    }
                    if (state == 3 && subState == 0) {
                        printf("Airline is %s\n", token);
                        break;
                    }
                }
            }
            fclose(file);
        }

    }

    closedir(dr);

}