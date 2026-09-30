# Kubernetes

Two ways to test the GPUs of a node from a cluster.

[`job.yaml`](job.yaml) is one Job that asks for every GPU of a node, runs a
memory pattern test on all of them, and fails when a workload fails on a card:

```bash
kubectl apply -f job.yaml
kubectl logs -f job/pantheon-node-check
```

Set `spec.template.spec.nodeName` to test a particular node, and
`nvidia.com/gpu` to the number of cards on it, so that the Job takes the
whole node and no other pod shares a card during the test.

[`pantheon-node-check/`](pantheon-node-check/) is a Helm chart that makes one
such Job for each node you name:

```bash
helm upgrade --install gpu-check ./pantheon-node-check \
    --set nodes='{gpu-node-01,gpu-node-02}' --set gpusPerNode=8
kubectl get jobs -l app.kubernetes.io/instance=gpu-check
```

A Job that completed means every workload passed on every card of that node.
A Job that failed means a workload failed on a card, and its log says which.
Give `reports.persistentVolumeClaim` a claim to keep the reports; an
`emptyDir` is gone with the pod.

Both need the NVIDIA device plugin (or the GPU operator) on the cluster, so
that `nvidia.com/gpu` can be requested. The chart was checked with `helm lint`
and the rendered manifests validated against the Kubernetes API schema; it
has not yet run on a cluster with GPUs.
