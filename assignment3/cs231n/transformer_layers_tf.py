import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import math
"""
My implementation of transformer layers based on TensorFlow 2.7.0
"""
class PositionalEncoding(layers.Layer):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = layers.Dropout(rate=dropout)
        assert embed_dim % 2 == 0
        self.pe = np.zeros((1, max_len, embed_dim))
        pos = np.arange(0, max_len).reshape(-1, 1) # (max_len, 1)
        # PE(pos,2i) = sin(pos/10000^(2i/embed_dim)); PE(pos,2i+1) = cos(pos*t)
        t = np.power(1e-4, np.arange(0, embed_dim, 2)/embed_dim)
        self.pe[:, :, 0::2] = np.sin(pos * t)
        self.pe[:, :, 1::2] = np.cos(pos * t)

    def call(self, x):
        N, S, D = x.shape
        output = np.zeros((N, S, D))
        output = x + self.pe[:, :S, :]
        output = self.dropout(output)
        return output


class MultiHeadAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.key = layers.Dense(embed_dim)
        self.query = layers.Dense(embed_dim)
        self.value = layers.Dense(embed_dim)
        self.proj = layers.Dense(embed_dim)
        self.H = num_heads
        self.softmax = layers.Softmax() # call with attention_mask of same shape
        self.dropout = layers.Dropout(rate=dropout)
        
    def call(self, query, key, value, attn_mask=None):
        N, S, D = query.shape
        N, T, D = value.shape
        output = np.zeros((N, T, D))
        q = tf.transpose(tf.reshape(self.query(query), (N, S, self.H, D//self.H)), [0, 2, 1, 3])
        k = tf.transpose(tf.reshape(self.key(key), (N, S, self.H, D//self.H)), [0, 2, 1, 3])
        v = tf.transpose(tf.reshape(self.value(value), (N, S, self.H, D//self.H)), [0, 2, 1, 3])
        e = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(D // self.H, dtype=tf.float32))
        a = self.softmax(e, attn_mask) # attention probs
        output = self.dropout(a) # dropout some probs
        output = tf.matmul(output, v) # (N, H, S, D/H)
        output = tf.reshape(tf.transpose(output, [0, 2, 1, 3]), (N, S, -1))  # (N, S, D)
        output = self.proj(output)
        return output