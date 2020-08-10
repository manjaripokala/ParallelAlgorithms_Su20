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

// Structure to represent a vertex and its distance
struct distNode {
    int64_t node;
    int64_t dist;

    bool operator<(const distNode &rhs) const {
        return dist > rhs.dist || (dist == rhs.dist && node > rhs.node);;
    }
};

// Structure to represent an edge
struct edge {
    int64_t from;
    int64_t to;
    int64_t weight;

    bool operator<(const edge &rhs) const {
        return weight > rhs.weight || (weight == rhs.weight && to > rhs.to);
    }
};

// Structure to represent a edge source & destination
struct fromTo {
    int64_t from;
    int64_t to;

    bool operator<(const fromTo &rhs) const {
        return to < rhs.to || (to == rhs.to && from < rhs.from);
    }
};

const char *get_filename_ext(const char *filename) {
    const char *dot = strrchr(filename, '.');
    if (!dot || dot == filename) return "";
    return dot + 1;
}

//static const char *JSON_STRING =
//        "{\"user\": \"johndoe\", \"admin\": false, \"uid\": 1000,\n  "
//        "\"groups\": [\"users\", \"wheel\", \"audio\", \"video\"]}";

static int jsoneq(const char *json, jsmntok_t *tok, const char *s) {
    if (tok->type == JSMN_STRING && (int) strlen(s) == tok->end - tok->start &&
        strncmp(json + tok->start, s, tok->end - tok->start) == 0) {
        return 0;
    }
    return -1;
}

int main() {
    struct dirent *de;  // Pointer for directory entry

    // opendir() returns a pointer of DIR type.
    DIR *dr = opendir("data");

    if (dr == NULL)  // opendir returns NULL if couldn't open directory
    {
        printf("Could not open current directory\n");
        return 0;
    }

    // Refer http://pubs.opengroup.org/onlinepubs/7990989775/xsh/readdir.html
    // for readdir()

    while ((de = readdir(dr)) != NULL) {
        char filePath[PATH_MAX + 1];
        if (strncmp(get_filename_ext(de->d_name), "json", 2) == 0) {
            realpath(de->d_name, filePath);
            printf("%s\n", filePath);

            FILE *f = fopen(filePath, "r");
            fseek(f, 0, SEEK_END);
            long fsize = ftell(f);
            fseek(f, 0, SEEK_SET);  /* same as rewind(f); */

            char *JSON_STRING = (char *) malloc(fsize + 1);
            fread(JSON_STRING, 1, fsize, f);
            printf("made it");
            fclose(f);
            JSON_STRING[fsize] = 0;
            printf("%s\n", JSON_STRING);
//            int i;
//            int r;
//            jsmn_parser p;
//            jsmntok_t t[1000]; /* We expect no more than 128 tokens */
//
//            jsmn_init(&p);
//            r = jsmn_parse(&p, JSON_STRING, strlen(JSON_STRING), t,
//                           sizeof(t) / sizeof(t[0]));
//            if (r < 0) {
//                printf("Failed to parse JSON: %d\n", r);
//                return 1;
//            }
//
//            /* Assume the top-level element is an object */
//            if (r < 1 || t[0].type != JSMN_OBJECT) {
//                printf("Object expected\n");
//                return 1;
//            }
//
//            /* Loop over all keys of the root object */
//            for (i = 1; i < r; i++) {
//
//                if (jsoneq(JSON_STRING, &t[i], "nodes") == 0) {
//                    /* We may use strndup() to fetch string value */
//                    int j;
//                    printf("- Nodes:\n");
//                    if (t[i + 1].type != JSMN_ARRAY) {
//                        continue; /* We expect groups to be an array of strings */
//                    }
//                    for (j = 0; j < t[i + 1].size; j++) {
//                        jsmntok_t *g = &t[i + j + 2];
//                        printf("  * %.*s\n", g->end - g->start, JSON_STRING + g->start);
//                    }
//                    i += t[i + 1].size + 1;
////                printf("- Node: %.*s\n", t[i + 1].end - t[i + 1].start,
////                       JSON_STRING + t[i + 1].start);
////                i++;
//                } else if (jsoneq(JSON_STRING, &t[i], "edges") == 0) {
//                    int j;
//                    printf("- Edges:\n");
//                    if (t[i + 1].type != JSMN_ARRAY) {
//                        continue; /* We expect groups to be an array of strings */
//                    }
//                    for (j = 0; j < t[i + 1].size; j++) {
//                        jsmntok_t *g = &t[i + j + 2];
//                        printf("  * %.*s\n", g->end - g->start, JSON_STRING + g->start);
//                    }
//                    i += t[i + 1].size + 1;
//                } else {
//                    printf("Unexpected key: %.*s\n", t[i].end - t[i].start,
//                           JSON_STRING + t[i].start);
//                }
//            }
            return EXIT_SUCCESS;
        }
    }
    closedir(dr);


}