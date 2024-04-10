#include "Support/Graph.h"
#include <llvm/ADT/Hashing.h>
#include <mlir/IR/Value.h>
#include <fstream>
#include <iostream>

Graph::Node& Graph::add_node(std::string label, mlir::Value value) {
    nodes.try_emplace(mlir::hash_value(value), Graph::Node{id++, label, std::vector<Graph::Node>()});
    return nodes.find(mlir::hash_value(value))->second;
}

//template <typename T>
/*
void Graph::add_edge(std::string label, mlir::Value from, mlir::Value to) {
    // insert our two keys
    // eugh
    value_to_id.try_emplace(mlir::hash_value(from), id++);
    value_to_id.try_emplace(mlir::hash_value(from), id++);
    // this isn't gross because we guarantee that it's already a value
    struct Edge edge{.data = label, .from = value_to_id.find(mlir::hash_value(from))->second, .to = value_to_id.find(mlir::hash_value(to))->second};
    edges.push_back(edge);
}
*/
//template <typename T>
void Graph::to_file(const std::string& filename) {
    std::string sep = directed ? " -> " : " -- ";

    // is this the best way to do things?
    // TODO: maybe use a pipe
    std::ofstream file;
    file.open(filename);

    // write the file header
    file << '\n' << (directed ? "digraph" : "graph") << " generated_graph {\n";

    // write the nodes
    for (auto const& [_, node] : nodes) {
        file << '\t' << node.id << " [label=\"" << node.data << "\"];\n";
    }

    file << '\n';

    // write the edges
    for (auto const& [_, node] : nodes) {
        std::cout << "next!" << node.children.size();
        for (auto const& other : node.children) {
            file << '\t'<< node.id << sep << other.id << ";\n";
        }
    }

    // write the footer
    file << "}\n";

    // clean up
    file.close();
}

Graph Graph::from_file(const std::string& filename) {
    return Graph();
}
