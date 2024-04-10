#ifndef GRAPH_H
#define GRAPH_H

#include <llvm/ADT/Hashing.h>
#include <mlir/IR/Value.h>
#include <string>
#include <vector>

//template <typename T>
class Graph {
public:
    Graph(bool directed = true) : directed(directed) {}
    void add_edge(std::string label, mlir::Value from, mlir::Value to);
    //void add_edge(std::string label, T from, T to);
    void to_file(const std::string& filename);

    struct Node {
        const int id;
        std::string data;
        std::vector<Node> children;
    };

    Graph::Node& add_node(std::string label, mlir::Value value);
private:
    bool directed;
    int id = 0;
    // uhhh
    // gives a list of
    std::map<llvm::hash_code, Node> nodes;

    /*struct Edge {
        std::string data;
        Node from;
        Node to;
    };*/
    //std::vector<struct Edge> edges;
};

#endif
