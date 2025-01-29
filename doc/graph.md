<!--
    Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
    Licensed under the MIT License.
 -->

```
type = class onnxruntime::Graph {
  private:
    const onnxruntime::Model &owning_model_;
    onnx::GraphProto *graph_proto_;
    onnx::GraphProto deserialized_proto_data_;
    onnxruntime::InitializedTensorSet name_to_initial_tensor_;
    std::unordered_set<std::reference_wrapper<std::string const>, std::hash<std::string>, std::equal_to<std::string>, std::allocator<std::reference_wrapper<std::string const> > > sparse_tensor_names_;
    std::unique_ptr<onnxruntime::RuntimeOptimizationRecordContainer> runtime_optimizations_ptr_;
    onnxruntime::RuntimeOptimizationRecordContainer &runtime_optimizations_;
    onnxruntime::Graph::RuntimeOptimizationReplayContext runtime_optimization_replay_context_;
    onnxruntime::IOnnxRuntimeOpSchemaCollectionPtr schema_registry_;
    onnxruntime::InlinedVector fused_schemas_containers_;
    std::vector<std::unique_ptr<onnxruntime::Node>> nodes_;
    onnxruntime::GraphNodes iterable_nodes_;
    int num_of_nodes_;
    bool graph_resolve_needed_;
    bool graph_proto_sync_needed_;
    std::vector<unsigned long> nodes_in_topological_order_;
    std::vector<const onnxruntime::NodeArg*> graph_inputs_including_initializers_;
    bool graph_inputs_manually_set_;
    std::vector<const onnxruntime::NodeArg*> graph_inputs_excluding_initializers_;
    std::vector<const onnxruntime::NodeArg*> graph_overridable_initializers_;
    std::vector<const onnxruntime::NodeArg*> graph_outputs_;
    bool graph_outputs_manually_set_;
    std::unordered_set<const onnxruntime::NodeArg*> value_info_;
    std::unordered_map<std::string, std::unique_ptr<onnxruntime::NodeArg>> node_args_;
    int name_generator_;
    std::unordered_set<std::string> generated_node_names_;
    std::unordered_set<std::string> generated_node_arg_names_;
    std::unordered_map<std::string, unsigned long> node_arg_to_producer_node_;
    std::unordered_map<std::string, std::unordered_set<unsigned long>> node_arg_to_consumer_nodes_;
    std::unordered_map<std::string, int> domain_to_version_;
    onnxruntime::Version ir_version_;
    onnxruntime::Graph::ResolveContext resolve_context_;
    onnxruntime::Graph *parent_graph_;
    const onnxruntime::Node *parent_node_;
    std::unordered_set<std::string> outer_scope_node_arg_names_;
    int num_resolves_;
    const onnxruntime::logging::Logger &logger_;
    const bool strict_shape_type_inference_;
    const bool is_loaded_from_model_file_;

  public:
    const std::string & Name(void) const;
    const std::string & Description(void) const;
    const onnxruntime::Path & ModelPath(void) const;
    bool IsSubgraph(void) const;
    const onnxruntime::Graph * ParentGraph(void) const;
    onnxruntime::Graph * MutableParentGraph(void);
    bool StrictShapeTypeInference(void) const;
    void SetName(const std::string &);
    void SetDescription(const std::string &);
    onnxruntime::common::Status ReplaceInitializedTensor(onnx::TensorProto);
    onnxruntime::common::Status InjectExternalInitializedTensors(const onnxruntime::InlinedHashMap<std::string, OrtValue, std::allocator<std::pair<std::string const, OrtValue> > > &);
    void AddInitializedTensor(const onnx::TensorProto &);
    void RemoveInitializedTensor(const std::string &);
    bool IsInitializedTensor(const std::string &) const;
    bool IsSparseInitializer(const std::string &) const;
    bool GetInitializedTensor(const std::string &, const onnx::TensorProto *&) const;
    const onnxruntime::InitializedTensorSet & GetAllInitializedTensors(void) const;
    void CleanAllInitializedTensors(void);
    bool CanOverrideInitializer(void) const;
    const onnx::TensorProto * GetConstantInitializer(const std::string &, bool) const;
    const onnx::TensorProto * GetInitializer(const std::string &, bool) const;
    const std::vector<const onnxruntime::NodeArg*> & GetInputs(void) const;
    const std::vector<const onnxruntime::NodeArg*> & GetInputsIncludingInitializers(void) const;
    bool IsInputsIncludingInitializers(const onnxruntime::NodeArg *) const;
    const std::vector<const onnxruntime::NodeArg*> & GetOverridableInitializers(void) const;
    const std::vector<const onnxruntime::NodeArg*> & GetOutputs(void) const;
    bool IsOutput(const onnxruntime::NodeArg *) const;
    bool NodeProducesGraphOutput(const onnxruntime::Node &) const;
    std::vector<int> GetNodeOutputsInGraphOutputs(const onnxruntime::Node &) const;
    const std::unordered_set<const onnxruntime::NodeArg*> & GetValueInfo(void) const;
    void AddValueInfo(const onnxruntime::NodeArg *);
    const onnxruntime::Node * GetNode(onnxruntime::NodeIndex) const;
    onnxruntime::Node * GetNode(onnxruntime::NodeIndex);
    onnxruntime::GraphNodes & Nodes(void);
    const onnxruntime::GraphNodes & Nodes(void) const;
    onnxruntime::ConstGraphNodes FilteredNodes(onnxruntime::ValidNodes<std::vector<std::unique_ptr<onnxruntime::Node>> >::NodeFilterFunc &&) const;
    int MaxNodeIndex(void) const;
    int NumberOfNodes(void) const;
    onnxruntime::NodeArg * GetNodeArg(const std::string &);
    const onnxruntime::NodeArg * GetNodeArg(const std::string &) const;
    onnxruntime::NodeArg * GetNodeArgIncludingParentGraphs(const std::string &);
    onnxruntime::NodeArg & GetOrCreateNodeArg(const std::string &, const onnx::TypeProto *);
    std::string GenerateNodeArgName(const std::string &);
    std::string GenerateNodeName(const std::string &);
    onnxruntime::Node & AddNode(const onnxruntime::Node &);
    onnxruntime::Node & AddNode(const std::string &, const std::string &, const std::string &, gsl::span<onnxruntime::NodeArg* const>, gsl::span<onnxruntime::NodeArg* const>, const onnxruntime::NodeAttributes *, const std::string &);
    onnxruntime::Node & AddNode(const std::string &, const std::string &, const std::string &, std::initializer_list<onnxruntime::NodeArg*>, std::initializer_list<onnxruntime::NodeArg*>, const onnxruntime::NodeAttributes *, const std::string &);
    onnxruntime::Node & AddNode(const std::string &, const std::string &, const std::string &, gsl::span<onnxruntime::NodeArg* const>, std::initializer_list<onnxruntime::NodeArg*>, const onnxruntime::NodeAttributes *, const std::string &);
    onnxruntime::Node & AddNode(const std::string &, const std::string &, const std::string &, std::initializer_list<onnxruntime::NodeArg*>, gsl::span<onnxruntime::NodeArg* const>, const onnxruntime::NodeAttributes *, const std::string &);
  private:
    onnxruntime::Node & AddNode(const onnx::NodeProto &, const onnxruntime::ArgNameToTypeMap &);
  public:
    bool RemoveNode(onnxruntime::NodeIndex);
    void AddEdge(onnxruntime::NodeIndex, onnxruntime::NodeIndex, int, int);
    void RemoveEdge(onnxruntime::NodeIndex, onnxruntime::NodeIndex, int, int);
    bool AddControlEdge(onnxruntime::NodeIndex, onnxruntime::NodeIndex);
    onnxruntime::Graph & SetGraphResolveNeeded(void);
    bool GraphResolveNeeded(void) const;
  private:
    onnxruntime::Graph & GraphResolveNeeded(bool);
  public:
    onnxruntime::Graph & SetGraphProtoSyncNeeded(void);
    bool GraphProtoSyncNeeded(void) const;
  private:
    onnxruntime::Graph & GraphProtoSyncNeeded(bool);
  public:
    void ReverseDFSFrom(gsl::span<unsigned long const>, const std::function<void(const onnxruntime::Node*)> &, const std::function<void(const onnxruntime::Node*)> &, const std::function<bool(const onnxruntime::Node*, const onnxruntime::Node*)> &) const;
    void ReverseDFSFrom(gsl::span<onnxruntime::Node const* const>, const std::function<void(const onnxruntime::Node*)> &, const std::function<void(const onnxruntime::Node*)> &, const std::function<bool(const onnxruntime::Node*, const onnxruntime::Node*)> &) const;
    void ReverseDFSFrom(gsl::span<onnxruntime::Node const* const>, const std::function<void(const onnxruntime::Node*)> &, const std::function<void(const onnxruntime::Node*)> &, const std::function<bool(const onnxruntime::Node*, const onnxruntime::Node*)> &, const std::function<bool(const onnxruntime::Node*, const o
nnxruntime::Node*)> &) const;
    void KahnsTopologicalSort(const std::function<void(const onnxruntime::Node*)> &, const std::function<bool(const onnxruntime::Node*, const onnxruntime::Node*)> &) const;
    const std::unordered_map<std::string, int> & DomainToVersionMap(void) const;
    onnxruntime::Node & BeginFuseSubGraph(const onnxruntime::IndexedSubGraph &, const std::string &);
    void CancelFuseSubGraph(const onnxruntime::Node &);
    void FinalizeFuseSubGraph(const onnxruntime::IndexedSubGraph &, onnxruntime::Node &);
    const onnx::GraphProto & ToGraphProto(void);
    onnx::GraphProto ToGraphProto(void) const;
    onnx::GraphProto ToGraphProtoWithExternalInitializers(const std::string &, size_t) const;
    onnxruntime::IOnnxRuntimeOpSchemaCollectionPtr GetSchemaRegistry(void) const;
    bool SetOpSchemaFromRegistryForNode(onnxruntime::Node &);
    onnxruntime::Node & FuseSubGraph(const onnxruntime::IndexedSubGraph &, const std::string &);
    onnxruntime::common::Status InlineFunction(onnxruntime::Node &);
    void AddOuterScopeNodeArg(const std::string &);
    void SetInputs(gsl::span<onnxruntime::NodeArg const* const>);
    void SetInputs(std::initializer_list<onnxruntime::NodeArg const*>);
    const onnxruntime::Model & GetModel(void) const;
    const onnxruntime::logging::Logger & GetLogger(void) const;
    void SetOutputs(gsl::span<onnxruntime::NodeArg const* const>);
    void SetOutputs(std::initializer_list<onnxruntime::NodeArg const*>);
    void SetNodeArgType(onnxruntime::NodeArg &, const onnx::TypeProto &);
    const onnxruntime::Node * GetProducerNode(const std::string &) const;
    onnxruntime::Node * GetMutableProducerNode(const std::string &);
    void UpdateProducerNode(const std::string &, onnxruntime::NodeIndex);
    std::vector<const onnxruntime::Node*> GetConsumerNodes(const std::string &) const;
    void AddConsumerNode(const std::string &, onnxruntime::Node *);
    void RemoveConsumerNode(const std::string &, onnxruntime::Node *);
    std::vector<onnxruntime::Node*> GetMutableConsumerNodes(const std::string &);
    void UpdateConsumerNodes(const std::string &, gsl::span<onnxruntime::Node* const>);
    void UpdateConsumerNodes(const std::string &, std::initializer_list<onnxruntime::Node*>);
    onnxruntime::common::Status UpdateShapeInference(onnxruntime::Node &);
    onnxruntime::common::Status Resolve(const onnxruntime::Graph::ResolveOptions &);
    onnxruntime::common::Status Resolve(void);
    const std::unordered_set<std::string> & GetOuterScopeNodeArgNames(void) const;
    onnxruntime::common::Status SaveToOrtFormat(flatbuffers::FlatBufferBuilder &, flatbuffers::Offset<onnxruntime::fbs::Graph> &) const;
    const onnxruntime::Node * ParentNode(void) const;
    bool IsOuterScopeValue(const std::string &) const;
    Graph(onnxruntime::Graph &, const onnxruntime::Node &, onnx::GraphProto &);
    Graph(const onnxruntime::Model &, onnxruntime::IOnnxRuntimeOpSchemaCollectionPtr, onnx::GraphProto &, const std::unordered_map<std::string, int> &, const onnxruntime::logging::Logger &, bool);
    Graph(void);
    Graph(const onnxruntime::Model &, const std::unordered_map<std::string, int> &, onnxruntime::IOnnxRuntimeOpSchemaCollectionPtr, onnxruntime::Graph *, const onnxruntime::Node *, const onnxruntime::logging::Logger &, bool);
    Graph(const onnxruntime::Model &, onnx::GraphProto *, const std::unordered_map<std::string, int> &, onnxruntime::Version, onnxruntime::IOnnxRuntimeOpSchemaCollectionPtr, const onnxruntime::logging::Logger &, bool);
    Graph(const onnxruntime::Model &, onnx::GraphProto *, const std::unordered_map<std::string, int> &, onnxruntime::Version, onnxruntime::IOnnxRuntimeOpSchemaCollectionPtr, onnxruntime::Graph *, const onnxruntime::Node *, const onnxruntime::logging::Logger &, bool);
    Graph(const onnxruntime::Graph &);
    Graph(onnxruntime::Graph &&);
    ~Graph();
    static onnxruntime::common::Status LoadFromOrtFormat(const onnxruntime::fbs::Graph &, const onnxruntime::Model &, const std::unordered_map<std::string, int> &, onnxruntime::IOnnxRuntimeOpSchemaCollectionPtr, const onnxruntime::logging::Logger &, std::unique_ptr<onnxruntime::Graph> &);
    static onnxruntime::common::Status LoadFromOrtFormat(const onnxruntime::fbs::Graph &, onnxruntime::Graph &, const onnxruntime::Node &, const onnxruntime::logging::Logger &, std::unique_ptr<onnxruntime::Graph> &);
    onnxruntime::common::Status LoadFromOrtFormat(const onnxruntime::fbs::Graph &);
    const onnxruntime::RuntimeOptimizationRecordContainer & RuntimeOptimizations(void) const;
    onnxruntime::RuntimeOptimizationRecordContainer & MutableRuntimeOptimizations(void);
    const onnxruntime::Graph::RuntimeOptimizationReplayContext & RuntimeOptimizationReplayCtx(void) const;
    onnxruntime::Graph::RuntimeOptimizationReplayContext & MutableRuntimeOptimizationReplayCtx(void);
    onnxruntime::Graph & operator=(const onnxruntime::Graph &);
    onnxruntime::Graph & operator=(onnxruntime::Graph &&);
}
```
