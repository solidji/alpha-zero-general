# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:55:58 2017

@author: XuGang
"""
from __future__ import print_function
from __future__ import absolute_import
# from .gameutil import card_show, choose, game_init
import numpy as np

# from rl.init_model import model_init

action_dict = {
    '[1, 1, 1, 10, 10]': 205,
    '[1, 1, 1, 10]': 49,
    '[1, 1, 1, 11, 11]': 206,
    '[1, 1, 1, 11]': 50,
    '[1, 1, 1, 12, 12]': 207,
    '[1, 1, 1, 12]': 51,
    '[1, 1, 1, 13, 13]': 208,
    '[1, 1, 1, 13]': 52,
    '[1, 1, 1, 14]': 416,
    '[1, 1, 1, 15]': 403,
    '[1, 1, 1, 1]': 353,
    '[1, 1, 1, 2, 2]': 197,
    '[1, 1, 1, 2]': 41,
    '[1, 1, 1, 3, 3]': 198,
    '[1, 1, 1, 3]': 42,
    '[1, 1, 1, 4, 4]': 199,
    '[1, 1, 1, 4]': 43,
    '[1, 1, 1, 5, 5]': 200,
    '[1, 1, 1, 5]': 44,
    '[1, 1, 1, 6, 6]': 201,
    '[1, 1, 1, 6]': 45,
    '[1, 1, 1, 7, 7]': 202,
    '[1, 1, 1, 7]': 46,
    '[1, 1, 1, 8, 8]': 203,
    '[1, 1, 1, 8]': 47,
    '[1, 1, 1, 9, 9]': 204,
    '[1, 1, 1, 9]': 48,
    '[1, 1, 10, 10, 10]': 305,
    '[1, 1, 11, 11, 11]': 317,
    '[1, 1, 12, 12, 12]': 329,
    '[1, 1, 13, 13, 13]': 341,
    '[1, 1, 1]': 28,
    '[1, 1, 2, 2, 2]': 209,
    '[1, 1, 3, 3, 3]': 221,
    '[1, 1, 4, 4, 4]': 233,
    '[1, 1, 5, 5, 5]': 245,
    '[1, 1, 6, 6, 6]': 257,
    '[1, 1, 7, 7, 7]': 269,
    '[1, 1, 8, 8, 8]': 281,
    '[1, 1, 9, 9, 9]': 293,
    '[1, 10, 10, 10]': 149,
    '[1, 10, 11, 12, 13]': 374,
    '[1, 11, 11, 11]': 161,
    '[1, 12, 12, 12]': 173,
    '[1, 13, 13, 13]': 185,
    '[1, 1]': 15,
    '[1, 2, 2, 2]': 53,
    '[1, 3, 3, 3]': 65,
    '[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]': 402,
    '[1, 4, 4, 4]': 77,
    '[1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]': 401,
    '[1, 5, 5, 5]': 89,
    '[1, 5, 6, 7, 8, 9, 10, 11, 12, 13]': 399,
    '[1, 6, 6, 6]': 101,
    '[1, 6, 7, 8, 9, 10, 11, 12, 13]': 396,
    '[1, 7, 7, 7]': 113,
    '[1, 7, 8, 9, 10, 11, 12, 13]': 392,
    '[1, 8, 8, 8]': 125,
    '[1, 8, 9, 10, 11, 12, 13]': 387,
    '[1, 9, 10, 11, 12, 13]': 381,
    '[1, 9, 9, 9]': 137,
    '[10, 10, 10, 10]': 362,
    '[10, 10, 10, 11, 11]': 314,
    '[10, 10, 10, 11]': 158,
    '[10, 10, 10, 12, 12]': 315,
    '[10, 10, 10, 12]': 159,
    '[10, 10, 10, 13, 13]': 316,
    '[10, 10, 10, 13]': 160,
    '[10, 10, 10, 14]': 425,
    '[10, 10, 10, 15]': 412,
    '[10, 10, 10]': 37,
    '[10, 10, 11, 11, 11]': 326,
    '[10, 10, 12, 12, 12]': 338,
    '[10, 10, 13, 13, 13]': 350,
    '[10, 10]': 24,
    '[10, 11, 11, 11]': 170,
    '[10, 12, 12, 12]': 182,
    '[10, 13, 13, 13]': 194,
    '[10]': 9,
    '[11, 11, 11, 11]': 363,
    '[11, 11, 11, 12, 12]': 327,
    '[11, 11, 11, 12]': 171,
    '[11, 11, 11, 13, 13]': 328,
    '[11, 11, 11, 13]': 172,
    '[11, 11, 11, 14]': 426,
    '[11, 11, 11, 15]': 413,
    '[11, 11, 11]': 38,
    '[11, 11, 12, 12, 12]': 339,
    '[11, 11, 13, 13, 13]': 351,
    '[11, 11]': 25,
    '[11, 12, 12, 12]': 183,
    '[11, 13, 13, 13]': 195,
    '[11]': 10,
    '[12, 12, 12, 12]': 364,
    '[12, 12, 12, 13, 13]': 340,
    '[12, 12, 12, 13]': 184,
    '[12, 12, 12, 14]': 427,
    '[12, 12, 12, 15]': 414,
    '[12, 12, 12]': 39,
    '[12, 12, 13, 13, 13]': 352,
    '[12, 12]': 26,
    '[12, 13, 13, 13]': 196,
    '[12]': 11,
    '[13, 13, 13, 13]': 365,
    '[13, 13, 13, 14]': 428,
    '[13, 13, 13, 15]': 415,
    '[13, 13, 13]': 40,
    '[13, 13]': 27,
    '[13]': 12,
    '[14, 15]': 366,
    '[14]': 13,
    '[15]': 14,
    '[1]': 0,
    '[2, 10, 10, 10]': 150,
    '[2, 11, 11, 11]': 162,
    '[2, 12, 12, 12]': 174,
    '[2, 13, 13, 13]': 186,
    '[2, 2, 10, 10, 10]': 306,
    '[2, 2, 11, 11, 11]': 318,
    '[2, 2, 12, 12, 12]': 330,
    '[2, 2, 13, 13, 13]': 342,
    '[2, 2, 2, 10, 10]': 217,
    '[2, 2, 2, 10]': 61,
    '[2, 2, 2, 11, 11]': 218,
    '[2, 2, 2, 11]': 62,
    '[2, 2, 2, 12, 12]': 219,
    '[2, 2, 2, 12]': 63,
    '[2, 2, 2, 13, 13]': 220,
    '[2, 2, 2, 13]': 64,
    '[2, 2, 2, 14]': 417,
    '[2, 2, 2, 15]': 404,
    '[2, 2, 2, 2]': 354,
    '[2, 2, 2, 3, 3]': 210,
    '[2, 2, 2, 3]': 54,
    '[2, 2, 2, 4, 4]': 211,
    '[2, 2, 2, 4]': 55,
    '[2, 2, 2, 5, 5]': 212,
    '[2, 2, 2, 5]': 56,
    '[2, 2, 2, 6, 6]': 213,
    '[2, 2, 2, 6]': 57,
    '[2, 2, 2, 7, 7]': 214,
    '[2, 2, 2, 7]': 58,
    '[2, 2, 2, 8, 8]': 215,
    '[2, 2, 2, 8]': 59,
    '[2, 2, 2, 9, 9]': 216,
    '[2, 2, 2, 9]': 60,
    '[2, 2, 2]': 29,
    '[2, 2, 3, 3, 3]': 222,
    '[2, 2, 4, 4, 4]': 234,
    '[2, 2, 5, 5, 5]': 246,
    '[2, 2, 6, 6, 6]': 258,
    '[2, 2, 7, 7, 7]': 270,
    '[2, 2, 8, 8, 8]': 282,
    '[2, 2, 9, 9, 9]': 294,
    '[2, 2]': 16,
    '[2, 3, 3, 3]': 66,
    '[2, 4, 4, 4]': 78,
    '[2, 5, 5, 5]': 90,
    '[2, 6, 6, 6]': 102,
    '[2, 7, 7, 7]': 114,
    '[2, 8, 8, 8]': 126,
    '[2, 9, 9, 9]': 138,
    '[2]': 1,
    '[3, 10, 10, 10]': 151,
    '[3, 11, 11, 11]': 163,
    '[3, 12, 12, 12]': 175,
    '[3, 13, 13, 13]': 187,
    '[3, 3, 10, 10, 10]': 307,
    '[3, 3, 11, 11, 11]': 319,
    '[3, 3, 12, 12, 12]': 331,
    '[3, 3, 13, 13, 13]': 343,
    '[3, 3, 3, 10, 10]': 229,
    '[3, 3, 3, 10]': 73,
    '[3, 3, 3, 11, 11]': 230,
    '[3, 3, 3, 11]': 74,
    '[3, 3, 3, 12, 12]': 231,
    '[3, 3, 3, 12]': 75,
    '[3, 3, 3, 13, 13]': 232,
    '[3, 3, 3, 13]': 76,
    '[3, 3, 3, 14]': 418,
    '[3, 3, 3, 15]': 405,
    '[3, 3, 3, 3]': 355,
    '[3, 3, 3, 4, 4]': 223,
    '[3, 3, 3, 4]': 67,
    '[3, 3, 3, 5, 5]': 224,
    '[3, 3, 3, 5]': 68,
    '[3, 3, 3, 6, 6]': 225,
    '[3, 3, 3, 6]': 69,
    '[3, 3, 3, 7, 7]': 226,
    '[3, 3, 3, 7]': 70,
    '[3, 3, 3, 8, 8]': 227,
    '[3, 3, 3, 8]': 71,
    '[3, 3, 3, 9, 9]': 228,
    '[3, 3, 3, 9]': 72,
    '[3, 3, 3]': 30,
    '[3, 3, 4, 4, 4]': 235,
    '[3, 3, 5, 5, 5]': 247,
    '[3, 3, 6, 6, 6]': 259,
    '[3, 3, 7, 7, 7]': 271,
    '[3, 3, 8, 8, 8]': 283,
    '[3, 3, 9, 9, 9]': 295,
    '[3, 3]': 17,
    '[3, 4, 4, 4]': 79,
    '[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]': 400,
    '[3, 4, 5, 6, 7, 8, 9, 10, 11, 12]': 397,
    '[3, 4, 5, 6, 7, 8, 9, 10, 11]': 393,
    '[3, 4, 5, 6, 7, 8, 9, 10]': 388,
    '[3, 4, 5, 6, 7, 8, 9]': 382,
    '[3, 4, 5, 6, 7, 8]': 375,
    '[3, 4, 5, 6, 7]': 367,
    '[3, 5, 5, 5]': 91,
    '[3, 6, 6, 6]': 103,
    '[3, 7, 7, 7]': 115,
    '[3, 8, 8, 8]': 127,
    '[3, 9, 9, 9]': 139,
    '[3]': 2,
    '[4, 10, 10, 10]': 152,
    '[4, 11, 11, 11]': 164,
    '[4, 12, 12, 12]': 176,
    '[4, 13, 13, 13]': 188,
    '[4, 4, 10, 10, 10]': 308,
    '[4, 4, 11, 11, 11]': 320,
    '[4, 4, 12, 12, 12]': 332,
    '[4, 4, 13, 13, 13]': 344,
    '[4, 4, 4, 10, 10]': 241,
    '[4, 4, 4, 10]': 85,
    '[4, 4, 4, 11, 11]': 242,
    '[4, 4, 4, 11]': 86,
    '[4, 4, 4, 12, 12]': 243,
    '[4, 4, 4, 12]': 87,
    '[4, 4, 4, 13, 13]': 244,
    '[4, 4, 4, 13]': 88,
    '[4, 4, 4, 14]': 419,
    '[4, 4, 4, 15]': 406,
    '[4, 4, 4, 4]': 356,
    '[4, 4, 4, 5, 5]': 236,
    '[4, 4, 4, 5]': 80,
    '[4, 4, 4, 6, 6]': 237,
    '[4, 4, 4, 6]': 81,
    '[4, 4, 4, 7, 7]': 238,
    '[4, 4, 4, 7]': 82,
    '[4, 4, 4, 8, 8]': 239,
    '[4, 4, 4, 8]': 83,
    '[4, 4, 4, 9, 9]': 240,
    '[4, 4, 4, 9]': 84,
    '[4, 4, 4]': 31,
    '[4, 4, 5, 5, 5]': 248,
    '[4, 4, 6, 6, 6]': 260,
    '[4, 4, 7, 7, 7]': 272,
    '[4, 4, 8, 8, 8]': 284,
    '[4, 4, 9, 9, 9]': 296,
    '[4, 4]': 18,
    '[4, 5, 5, 5]': 92,
    '[4, 5, 6, 7, 8, 9, 10, 11, 12, 13]': 398,
    '[4, 5, 6, 7, 8, 9, 10, 11, 12]': 394,
    '[4, 5, 6, 7, 8, 9, 10, 11]': 389,
    '[4, 5, 6, 7, 8, 9, 10]': 383,
    '[4, 5, 6, 7, 8, 9]': 376,
    '[4, 5, 6, 7, 8]': 368,
    '[4, 6, 6, 6]': 104,
    '[4, 7, 7, 7]': 116,
    '[4, 8, 8, 8]': 128,
    '[4, 9, 9, 9]': 140,
    '[4]': 3,
    '[5, 10, 10, 10]': 153,
    '[5, 11, 11, 11]': 165,
    '[5, 12, 12, 12]': 177,
    '[5, 13, 13, 13]': 189,
    '[5, 5, 10, 10, 10]': 309,
    '[5, 5, 11, 11, 11]': 321,
    '[5, 5, 12, 12, 12]': 333,
    '[5, 5, 13, 13, 13]': 345,
    '[5, 5, 5, 10, 10]': 253,
    '[5, 5, 5, 10]': 97,
    '[5, 5, 5, 11, 11]': 254,
    '[5, 5, 5, 11]': 98,
    '[5, 5, 5, 12, 12]': 255,
    '[5, 5, 5, 12]': 99,
    '[5, 5, 5, 13, 13]': 256,
    '[5, 5, 5, 13]': 100,
    '[5, 5, 5, 14]': 420,
    '[5, 5, 5, 15]': 407,
    '[5, 5, 5, 5]': 357,
    '[5, 5, 5, 6, 6]': 249,
    '[5, 5, 5, 6]': 93,
    '[5, 5, 5, 7, 7]': 250,
    '[5, 5, 5, 7]': 94,
    '[5, 5, 5, 8, 8]': 251,
    '[5, 5, 5, 8]': 95,
    '[5, 5, 5, 9, 9]': 252,
    '[5, 5, 5, 9]': 96,
    '[5, 5, 5]': 32,
    '[5, 5, 6, 6, 6]': 261,
    '[5, 5, 7, 7, 7]': 273,
    '[5, 5, 8, 8, 8]': 285,
    '[5, 5, 9, 9, 9]': 297,
    '[5, 5]': 19,
    '[5, 6, 6, 6]': 105,
    '[5, 6, 7, 8, 9, 10, 11, 12, 13]': 395,
    '[5, 6, 7, 8, 9, 10, 11, 12]': 390,
    '[5, 6, 7, 8, 9, 10, 11]': 384,
    '[5, 6, 7, 8, 9, 10]': 377,
    '[5, 6, 7, 8, 9]': 369,
    '[5, 7, 7, 7]': 117,
    '[5, 8, 8, 8]': 129,
    '[5, 9, 9, 9]': 141,
    '[5]': 4,
    '[6, 10, 10, 10]': 154,
    '[6, 11, 11, 11]': 166,
    '[6, 12, 12, 12]': 178,
    '[6, 13, 13, 13]': 190,
    '[6, 6, 10, 10, 10]': 310,
    '[6, 6, 11, 11, 11]': 322,
    '[6, 6, 12, 12, 12]': 334,
    '[6, 6, 13, 13, 13]': 346,
    '[6, 6, 6, 10, 10]': 265,
    '[6, 6, 6, 10]': 109,
    '[6, 6, 6, 11, 11]': 266,
    '[6, 6, 6, 11]': 110,
    '[6, 6, 6, 12, 12]': 267,
    '[6, 6, 6, 12]': 111,
    '[6, 6, 6, 13, 13]': 268,
    '[6, 6, 6, 13]': 112,
    '[6, 6, 6, 14]': 421,
    '[6, 6, 6, 15]': 408,
    '[6, 6, 6, 6]': 358,
    '[6, 6, 6, 7, 7]': 262,
    '[6, 6, 6, 7]': 106,
    '[6, 6, 6, 8, 8]': 263,
    '[6, 6, 6, 8]': 107,
    '[6, 6, 6, 9, 9]': 264,
    '[6, 6, 6, 9]': 108,
    '[6, 6, 6]': 33,
    '[6, 6, 7, 7, 7]': 274,
    '[6, 6, 8, 8, 8]': 286,
    '[6, 6, 9, 9, 9]': 298,
    '[6, 6]': 20,
    '[6, 7, 7, 7]': 118,
    '[6, 7, 8, 9, 10, 11, 12, 13]': 391,
    '[6, 7, 8, 9, 10, 11, 12]': 385,
    '[6, 7, 8, 9, 10, 11]': 378,
    '[6, 7, 8, 9, 10]': 370,
    '[6, 8, 8, 8]': 130,
    '[6, 9, 9, 9]': 142,
    '[6]': 5,
    '[7, 10, 10, 10]': 155,
    '[7, 11, 11, 11]': 167,
    '[7, 12, 12, 12]': 179,
    '[7, 13, 13, 13]': 191,
    '[7, 7, 10, 10, 10]': 311,
    '[7, 7, 11, 11, 11]': 323,
    '[7, 7, 12, 12, 12]': 335,
    '[7, 7, 13, 13, 13]': 347,
    '[7, 7, 7, 10, 10]': 277,
    '[7, 7, 7, 10]': 121,
    '[7, 7, 7, 11, 11]': 278,
    '[7, 7, 7, 11]': 122,
    '[7, 7, 7, 12, 12]': 279,
    '[7, 7, 7, 12]': 123,
    '[7, 7, 7, 13, 13]': 280,
    '[7, 7, 7, 13]': 124,
    '[7, 7, 7, 14]': 422,
    '[7, 7, 7, 15]': 409,
    '[7, 7, 7, 7]': 359,
    '[7, 7, 7, 8, 8]': 275,
    '[7, 7, 7, 8]': 119,
    '[7, 7, 7, 9, 9]': 276,
    '[7, 7, 7, 9]': 120,
    '[7, 7, 7]': 34,
    '[7, 7, 8, 8, 8]': 287,
    '[7, 7, 9, 9, 9]': 299,
    '[7, 7]': 21,
    '[7, 8, 8, 8]': 131,
    '[7, 8, 9, 10, 11, 12, 13]': 386,
    '[7, 8, 9, 10, 11, 12]': 379,
    '[7, 8, 9, 10, 11]': 371,
    '[7, 9, 9, 9]': 143,
    '[7]': 6,
    '[8, 10, 10, 10]': 156,
    '[8, 11, 11, 11]': 168,
    '[8, 12, 12, 12]': 180,
    '[8, 13, 13, 13]': 192,
    '[8, 8, 10, 10, 10]': 312,
    '[8, 8, 11, 11, 11]': 324,
    '[8, 8, 12, 12, 12]': 336,
    '[8, 8, 13, 13, 13]': 348,
    '[8, 8, 8, 10, 10]': 289,
    '[8, 8, 8, 10]': 133,
    '[8, 8, 8, 11, 11]': 290,
    '[8, 8, 8, 11]': 134,
    '[8, 8, 8, 12, 12]': 291,
    '[8, 8, 8, 12]': 135,
    '[8, 8, 8, 13, 13]': 292,
    '[8, 8, 8, 13]': 136,
    '[8, 8, 8, 14]': 423,
    '[8, 8, 8, 15]': 410,
    '[8, 8, 8, 8]': 360,
    '[8, 8, 8, 9, 9]': 288,
    '[8, 8, 8, 9]': 132,
    '[8, 8, 8]': 35,
    '[8, 8, 9, 9, 9]': 300,
    '[8, 8]': 22,
    '[8, 9, 10, 11, 12, 13]': 380,
    '[8, 9, 10, 11, 12]': 372,
    '[8, 9, 9, 9]': 144,
    '[8]': 7,
    '[9, 10, 10, 10]': 157,
    '[9, 10, 11, 12, 13]': 373,
    '[9, 11, 11, 11]': 169,
    '[9, 12, 12, 12]': 181,
    '[9, 13, 13, 13]': 193,
    '[9, 9, 10, 10, 10]': 313,
    '[9, 9, 11, 11, 11]': 325,
    '[9, 9, 12, 12, 12]': 337,
    '[9, 9, 13, 13, 13]': 349,
    '[9, 9, 9, 10, 10]': 301,
    '[9, 9, 9, 10]': 145,
    '[9, 9, 9, 11, 11]': 302,
    '[9, 9, 9, 11]': 146,
    '[9, 9, 9, 12, 12]': 303,
    '[9, 9, 9, 12]': 147,
    '[9, 9, 9, 13, 13]': 304,
    '[9, 9, 9, 13]': 148,
    '[9, 9, 9, 14]': 424,
    '[9, 9, 9, 15]': 411,
    '[9, 9, 9, 9]': 361,
    '[9, 9, 9]': 36,
    '[9, 9]': 23,
    '[9]': 8
}


############################################
#                 游戏类                   #
############################################
class Poker(object):

    def __init__(self, models=["random", "random", "random"], my_config=None):
        # 初始化一副扑克牌类
        self.cards = Cards()

        # play相关参数
        self.end = False
        self.last_move_type = self.last_move = "start"
        self.playround = 1
        self.i = 0
        self.yaobuqis = []

        # choose模型
        self.models = models

        self.my_config = my_config
        self.actions_lookuptable = action_dict

    # 发牌
    def game_init(self, players, playrecords, cards, train):

        if train:
            # 洗牌
            np.random.shuffle(cards.cards)
            # 排序
            p1_cards = cards.cards[:18]
            p1_cards.sort(key=lambda x: x.rank)
            p2_cards = cards.cards[18:36]
            p2_cards.sort(key=lambda x: x.rank)
            p3_cards = cards.cards[36:]
            p3_cards.sort(key=lambda x: x.rank)
            players[0].cards_left = playrecords.cards_left1 = p1_cards
            players[1].cards_left = playrecords.cards_left2 = p2_cards
            players[2].cards_left = playrecords.cards_left3 = p3_cards
        else:
            # 洗牌
            np.random.shuffle(cards.cards)
            # 排序
            p1_cards = cards.cards[:20]
            p1_cards.sort(key=lambda x: x.rank)
            p2_cards = cards.cards[20:37]
            p2_cards.sort(key=lambda x: x.rank)
            p3_cards = cards.cards[37:]
            p3_cards.sort(key=lambda x: x.rank)
            players[0].cards_left = playrecords.cards_left1 = p1_cards
            players[1].cards_left = playrecords.cards_left2 = p2_cards
            players[2].cards_left = playrecords.cards_left3 = p3_cards

    # 初始化
    def game_start(self, train, RL=None):

        # 初始化players
        self.players = []
        self.players.append(Player(1, self.models[0], self.my_config, self, RL))
        self.players.append(Player(2, self.models[1], self.my_config, self, RL))
        self.players.append(Player(3, self.models[2], self.my_config, self, RL))

        # 初始化扑克牌记录类
        self.playrecords = PlayRecords()

        # 发牌
        self.game_init(self.players, self.playrecords, self.cards, train)

    # 返回扑克牌记录类
    def get_record(self):
        web_show = WebShow(self.playrecords)
        # return jsonpickle.encode(web_show, unpicklable=False)
        return web_show

    # 返回下次出牌列表
    def get_next_moves(self):
        next_move_types, next_moves = self.players[self.i].get_moves(self.last_move_type, self.last_move,
                                                                     self.playrecords)
        return next_move_types, next_moves

    # 游戏进行
    def get_next_move(self, action):
        while (self.i <= 2):
            if self.i != 0:
                self.get_next_moves()
            self.last_move_type, self.last_move, self.end, self.yaobuqi = self.players[self.i].play(self.last_move_type,
                                                                                                    self.last_move,
                                                                                                    self.playrecords,
                                                                                                    action)
            if self.yaobuqi:
                self.yaobuqis.append(self.i)
            else:
                self.yaobuqis = []
            # 都要不起
            if len(self.yaobuqis) == 2:
                self.yaobuqis = []
                self.last_move_type = self.last_move = "start"
            if self.end:
                self.playrecords.winner = self.i + 1
                break
            self.i = self.i + 1
        # 一轮结束
        self.playround = self.playround + 1
        self.i = 0
        return self.playrecords.winner, self.end

    def execute_move(self, action, player):
        # 需要补检查action是否valid，player是否0-2
        self.i = player
        self.last_move_type, self.last_move, self.end, self.yaobuqi = \
            self.players[self.i].play(self.last_move_type, self.last_move, self.playrecords, action)
        if self.yaobuqi:
            self.yaobuqis.append(self.i)
        else:
            self.yaobuqis = []
        # 都要不起
        if len(self.yaobuqis) == 2:
            self.yaobuqis = []
            self.last_move_type = self.last_move = "start"
        if self.end:
            self.playrecords.winner = self.i + 1
        self.i = self.i + 1

        return

############################################
#              扑克牌相关类                 #
############################################
class Cards(object):
    """
    一副扑克牌类,54张排,abcd四种花色,小王14-a,大王15-a
    """

    def __init__(self):
        # 初始化扑克牌类型
        self.cards_type = ['1-a-12', '1-b-12', '1-c-12', '1-d-12',
                           '2-a-13', '2-b-13', '2-c-13', '2-d-13',
                           '3-a-1', '3-b-1', '3-c-1', '3-d-1',
                           '4-a-2', '4-b-2', '4-c-2', '4-d-2',
                           '5-a-3', '5-b-3', '5-c-3', '5-d-3',
                           '6-a-4', '6-b-4', '6-c-4', '6-d-4',
                           '7-a-5', '7-b-5', '7-c-5', '7-d-5',
                           '8-a-6', '8-b-6', '8-c-6', '8-d-6',
                           '9-a-7', '9-b-7', '9-c-7', '9-d-7',
                           '10-a-8', '10-b-8', '10-c-8', '10-d-8',
                           '11-a-9', '11-b-9', '11-c-9', '11-d-9',
                           '12-a-10', '12-b-10', '12-c-10', '12-d-10',
                           '13-a-11', '13-b-11', '13-c-11', '13-d-11',
                           '14-a-14', '15-a-15']
        # 初始化扑克牌类
        self.cards = self.get_cards()

    # 初始化扑克牌类
    def get_cards(self):
        cards = []
        for card_type in self.cards_type:
            cards.append(Card(card_type))
        # 打乱顺序
        # np.random.shuffle(cards)
        return cards


class Card(object):
    """
    扑克牌类
    """

    def __init__(self, card_type):
        self.card_type = card_type
        # 名称
        self.name = self.card_type.split('-')[0]
        # 花色
        self.color = self.card_type.split('-')[1]
        # 大小
        self.rank = int(self.card_type.split('-')[2])

    # 判断大小
    def bigger_than(self, card_instance):
        if (self.rank > card_instance.rank):
            return True
        else:
            return False


class PlayRecords(object):
    """
    扑克牌记录类
    """

    def __init__(self):
        # 当前手牌
        self.cards_left1 = []
        self.cards_left2 = []
        self.cards_left3 = []

        # 可能出牌选择
        self.next_moves1 = []
        self.next_moves2 = []
        self.next_moves3 = []

        # 出牌记录
        self.next_move1 = []
        self.next_move2 = []
        self.next_move3 = []

        # 出牌记录
        self.records = []

        # 胜利者
        # winner=0,1,2,3 0表示未结束,1,2,3表示winner
        self.winner = 0

        # 出牌者
        self.player = 1

    # 展示
    def show(self, info):
        print(info)
        card_show(self.cards_left1, "player 1", 1)
        card_show(self.cards_left2, "player 2", 1)
        card_show(self.cards_left3, "player 3", 1)
        # card_show(self.records, "record", 3)


############################################
#              出牌相关类                   #
############################################
class Moves(object):
    """
    出牌类,单,对,三,三带一,三带二,顺子,炸弹
    """

    def __init__(self):
        # 出牌信息
        self.dan = []
        self.dui = []
        self.san = []
        self.san_dai_yi = []
        self.san_dai_er = []
        self.bomb = []
        self.shunzi = []

        # 牌数量信息
        self.card_num_info = {}
        # 牌顺序信息,计算顺子
        self.card_order_info = []
        # 王牌信息
        self.king = []

        # 下次出牌
        self.next_moves = []
        # 下次出牌类型
        self.next_moves_type = []

    # 获取全部出牌列表
    def get_total_moves(self, cards_left):

        # 统计牌数量/顺序/王牌信息
        for i in cards_left:
            # 王牌信息
            if i.rank in [14, 15]:
                self.king.append(i)
            # 数量
            tmp = self.card_num_info.get(i.rank, [])
            if len(tmp) == 0:
                self.card_num_info[i.rank] = [i]
            else:
                self.card_num_info[i.rank].append(i)
            # 顺序
            if i.rank in [13, 14, 15]:  # 不统计2,小王,大王
                continue
            elif len(self.card_order_info) == 0:
                self.card_order_info.append(i)
            elif i.rank != self.card_order_info[-1].rank:
                self.card_order_info.append(i)

        # 王炸
        if len(self.king) == 2:
            self.bomb.append(self.king)

        # 出单,出对,出三,炸弹(考虑拆开)
        for k, v in self.card_num_info.items():
            if len(v) == 1:
                self.dan.append(v)
        for k, v in self.card_num_info.items():
            if len(v) == 2:
                self.dui.append(v)
                self.dan.append(v[:1])
        for k, v in self.card_num_info.items():
            if len(v) == 3:
                self.san.append(v)
                self.dui.append(v[:2])
                self.dan.append(v[:1])
        for k, v in self.card_num_info.items():
            if len(v) == 4:
                self.bomb.append(v)
                self.san.append(v[:3])
                self.dui.append(v[:2])
                self.dan.append(v[:1])

        # 三带一,三带二
        for san in self.san:
            # if self.dan[0][0].name != san[0].name:
            #    self.san_dai_yi.append(san+self.dan[0])
            # if self.dui[0][0].name != san[0].name:
            #    self.san_dai_er.append(san+self.dui[0])
            for dan in self.dan:
                # 防止重复
                if dan[0].name != san[0].name:
                    self.san_dai_yi.append(san + dan)
            for dui in self.dui:
                # 防止重复
                if dui[0].name != san[0].name:
                    self.san_dai_er.append(san + dui)

                    # 获取最长顺子
        max_len = []
        for i in self.card_order_info:
            if i == self.card_order_info[0]:
                max_len.append(i)
            elif max_len[-1].rank == i.rank - 1:
                max_len.append(i)
            else:
                if len(max_len) >= 5:
                    self.shunzi.append(max_len)
                max_len = [i]
        # 最后一轮
        if len(max_len) >= 5:
            self.shunzi.append(max_len)
            # 拆顺子
        shunzi_sub = []
        for i in self.shunzi:
            len_total = len(i)
            n = len_total - 5
            # 遍历所有可能顺子长度
            while (n > 0):
                len_sub = len_total - n
                j = 0
                while (len_sub + j <= len(i)):
                    # 遍历该长度所有组合
                    shunzi_sub.append(i[j:len_sub + j])
                    j = j + 1
                n = n - 1
        self.shunzi.extend(shunzi_sub)

    # 获取下次出牌列表
    def get_next_moves(self, last_move_type, last_move):
        # 没有last,全加上,除了bomb最后加
        if last_move_type == "start":
            moves_types = ["dan", "dui", "san", "san_dai_yi", "san_dai_er", "shunzi"]
            i = 0
            for move_type in [self.dan, self.dui, self.san, self.san_dai_yi,
                              self.san_dai_er, self.shunzi]:
                for move in move_type:
                    self.next_moves.append(move)
                    self.next_moves_type.append(moves_types[i])
                i = i + 1
        # 出单
        elif last_move_type == "dan":
            for move in self.dan:
                # 比last大
                if move[0].bigger_than(last_move[0]):
                    self.next_moves.append(move)
                    self.next_moves_type.append("dan")
        # 出对
        elif last_move_type == "dui":
            for move in self.dui:
                # 比last大
                if move[0].bigger_than(last_move[0]):
                    self.next_moves.append(move)
                    self.next_moves_type.append("dui")
        # 出三个
        elif last_move_type == "san":
            for move in self.san:
                # 比last大
                if move[0].bigger_than(last_move[0]):
                    self.next_moves.append(move)
                    self.next_moves_type.append("san")
        # 出三带一
        elif last_move_type == "san_dai_yi":
            for move in self.san_dai_yi:
                # 比last大
                if move[0].bigger_than(last_move[0]):
                    self.next_moves.append(move)
                    self.next_moves_type.append("san_dai_yi")
        # 出三带二
        elif last_move_type == "san_dai_er":
            for move in self.san_dai_er:
                # 比last大
                if move[0].bigger_than(last_move[0]):
                    self.next_moves.append(move)
                    self.next_moves_type.append("san_dai_er")
        # 出炸弹
        elif last_move_type == "bomb":
            for move in self.bomb:
                # 比last大
                if move[0].bigger_than(last_move[0]):
                    self.next_moves.append(move)
                    self.next_moves_type.append("bomb")
        # 出顺子
        elif last_move_type == "shunzi":
            for move in self.shunzi:
                # 相同长度
                if len(move) == len(last_move):
                    # 比last大
                    if move[0].bigger_than(last_move[0]):
                        self.next_moves.append(move)
                        self.next_moves_type.append("shunzi")
        else:
            print("last_move_type_wrong")

        # 除了bomb,都可以出炸
        if last_move_type != "bomb":
            for move in self.bomb:
                self.next_moves.append(move)
                self.next_moves_type.append("bomb")

        return self.next_moves_type, self.next_moves

    # 展示
    def show(self, info):
        print(info)
        # card_show(self.dan, "dan", 2)
        # card_show(self.dui, "dui", 2)
        # card_show(self.san, "san", 2)
        # card_show(self.san_dai_yi, "san_dai_yi", 2)
        # card_show(self.san_dai_er, "san_dai_er", 2)
        # card_show(self.bomb, "bomb", 2)
        # card_show(self.shunzi, "shunzi", 2)
        # card_show(self.next_moves, "next_moves", 2)


############################################
#              玩家相关类                   #
############################################
class Player(object):
    """
    player类
    """

    def __init__(self, player_id, model, my_config, game=None, RL=None):
        self.player_id = player_id
        self.cards_left = []
        # 出牌模式
        self.model = model
        # RL_model
        self.game = game
        self.my_config = my_config

        self.RL = RL

    # 展示
    def show(self, info):
        self.total_moves.show(info)
        card_show(self.next_move, "next_move", 1)
        # card_show(self.cards_left, "card_left", 1)

    # 根据next_move同步cards_left
    def record_move(self, playrecords):
        # 记录出牌者
        playrecords.player = self.player_id
        # playrecords中records记录[id,next_move]
        if self.next_move_type in ["yaobuqi", "buyao"]:
            self.next_move = self.next_move_type
            playrecords.records.append([self.player_id, self.next_move_type])
        else:
            playrecords.records.append([self.player_id, self.next_move])
            for i in self.next_move:
                self.cards_left.remove(i)
                # 同步playrecords
        if self.player_id == 1:
            playrecords.cards_left1 = self.cards_left
            playrecords.next_moves1.append(self.next_moves)
            playrecords.next_move1.append(self.next_move)
        elif self.player_id == 2:
            playrecords.cards_left2 = self.cards_left
            playrecords.next_moves2.append(self.next_moves)
            playrecords.next_move2.append(self.next_move)
        elif self.player_id == 3:
            playrecords.cards_left3 = self.cards_left
            playrecords.next_moves3.append(self.next_moves)
            playrecords.next_move3.append(self.next_move)
        # 是否牌局结束
        end = False
        if len(self.cards_left) == 0:
            end = True
        return end

    # 选牌
    def get_moves(self, last_move_type, last_move, playrecords):
        # 所有出牌可选列表
        self.total_moves = Moves()
        # 获取全部出牌列表
        self.total_moves.get_total_moves(self.cards_left)
        # 获取下次出牌列表
        self.next_move_types, self.next_moves = self.total_moves.get_next_moves(last_move_type, last_move)
        # 返回下次出牌列表
        return self.next_move_types, self.next_moves

    # 出牌
    def play(self, last_move_type, last_move, playrecords, action):
        # 主动调用一下，初始化self.next_move_type
        self.get_moves(last_move_type, last_move, playrecords)
        # 在next_moves中选择出牌方法
        self.next_move_type, self.next_move = choose_random(next_move_types=self.next_move_types,
                                                            next_moves=self.next_moves,
                                                            last_move_type=last_move_type)
        # 记录
        end = self.record_move(playrecords)
        # 展示
        # self.show("Player " + str(self.player_id))
        # 要不起&不要
        yaobuqi = False
        if self.next_move_type in ["yaobuqi", "buyao"]:
            yaobuqi = True
            self.next_move_type = last_move_type
            self.next_move = last_move

        return self.next_move_type, self.next_move, end, yaobuqi


############################################
#               网页展示类                 #
############################################
class WebShow(object):
    """
    网页展示类
    """

    def __init__(self, playrecords):

        # 胜利者
        self.winner = playrecords.winner

        # 剩余手牌
        self.cards_left1 = []
        for i in playrecords.cards_left1:
            self.cards_left1.append(i.name + i.color)
        self.cards_left2 = []
        for i in playrecords.cards_left2:
            self.cards_left2.append(i.name + i.color)
        self.cards_left3 = []
        for i in playrecords.cards_left3:
            self.cards_left3.append(i.name + i.color)

            # 可能出牌
        self.next_moves1 = []
        if len(playrecords.next_moves1) != 0:
            next_moves = playrecords.next_moves1[-1]
            for move in next_moves:
                cards = []
                for card in move:
                    cards.append(card.name + card.color)
                self.next_moves1.append(cards)
        self.next_moves2 = []
        if len(playrecords.next_moves2) != 0:
            next_moves = playrecords.next_moves2[-1]
            for move in next_moves:
                cards = []
                for card in move:
                    cards.append(card.name + card.color)
                self.next_moves2.append(cards)
        self.next_moves3 = []
        if len(playrecords.next_moves3) != 0:
            next_moves = playrecords.next_moves3[-1]
            for move in next_moves:
                cards = []
                for card in move:
                    cards.append(card.name + card.color)
                self.next_moves3.append(cards)

                # 出牌
        self.next_move1 = []
        if len(playrecords.next_move1) != 0:
            next_move = playrecords.next_move1[-1]
            if next_move in ["yaobuqi", "buyao"]:
                self.next_move1.append(next_move)
            else:
                for card in next_move:
                    self.next_move1.append(card.name + card.color)
        self.next_move2 = []
        if len(playrecords.next_move2) != 0:
            next_move = playrecords.next_move2[-1]
            if next_move in ["yaobuqi", "buyao"]:
                self.next_move2.append(next_move)
            else:
                for card in next_move:
                    self.next_move2.append(card.name + card.color)
        self.next_move3 = []
        if len(playrecords.next_move3) != 0:
            next_move = playrecords.next_move3[-1]
            if next_move in ["yaobuqi", "buyao"]:
                self.next_move3.append(next_move)
            else:
                for card in next_move:
                    self.next_move3.append(card.name + card.color)

                    # 记录
        self.records = []
        for i in playrecords.records:
            tmp = []
            tmp.append(i[0])
            tmp_name = []
            # 处理要不起
            try:
                for j in i[1]:
                    tmp_name.append(j.name + j.color)
                tmp.append(tmp_name)
            except:
                tmp.append(i[1])
            self.records.append(tmp)


############################################
#                   LR相关                 #
############################################
def get_state(playrecords, player):
    '''
    定义当前state为6x13的矩阵，分别是：
    自己手牌+自己打过的牌+下家打过的牌+上家打过的牌+下家上一手牌+上家上一手牌
    :param playrecords:
    :param player:
    :return:
    '''
    state = np.zeros((6, 15), dtype=int)

    for i in playrecords.cards_left1:
        state[0][i.rank - 1] += 1
    for cards in playrecords.next_move1:
        if cards in ["buyao", "yaobuqi"]:
            continue
        for card in cards:
            state[1][card.rank - 1] += 1
    for cards in playrecords.next_move2:
        if cards in ["buyao", "yaobuqi"]:
            continue
        for card in cards:
            state[2][card.rank - 1] += 1
    for cards in playrecords.next_move3:
        if cards in ["buyao", "yaobuqi"]:
            continue
        for card in cards:
            state[3][card.rank - 1] += 1
    if playrecords.next_move2:
        cards = playrecords.next_move2[-1]
        if cards not in ["buyao", "yaobuqi"]:
            for card in cards:
                state[4][card.rank - 1] += 1
    if playrecords.next_move3:
        cards = playrecords.next_move3[-1]
        if cards not in ["buyao", "yaobuqi"]:
            for card in cards:
                state[5][card.rank - 1] += 1
    #
    # # 手牌
    # if player == 1:
    #     cards_left = playrecords.cards_left1
    #     state[30] = len(playrecords.cards_left1)
    #     state[31] = len(playrecords.cards_left2)
    #     state[32] = len(playrecords.cards_left3)
    # elif player == 2:
    #     cards_left = playrecords.cards_left2
    #     state[30] = len(playrecords.cards_left2)
    #     state[31] = len(playrecords.cards_left3)
    #     state[32] = len(playrecords.cards_left1)
    # else:
    #     cards_left = playrecords.cards_left3
    #     state[30] = len(playrecords.cards_left3)
    #     state[31] = len(playrecords.cards_left1)
    #     state[32] = len(playrecords.cards_left2)
    # for i in cards_left:
    #     state[i.rank - 1] += 1
    # # 底牌
    # for cards in playrecords.records:
    #     if cards[1] in ["buyao", "yaobuqi"]:
    #         continue
    #     for card in cards[1]:
    #         state[card.rank - 1 + 15] += 1

    return state


def get_actions(next_moves, game):
    """
    0-14: 单出， 1-13，小王，大王
    15-27: 对，1-13
    28-40: 三，1-13
    41-196: 三带1，先遍历111.2，111.3，一直到131313.12
    197-352: 三带2，先遍历111.22,111.33,一直到131313.1212
    353-366: 炸弹，1111-13131313，加上王炸
    367-402: 先考虑5个的顺子，按照顺子开头从小到大进行编码，共计8+7+..+1=36
    430: yaobuqi
    429: buyao
    """
    actions_lookuptable = action_dict
    actions = []
    for cards in next_moves:
        key = []
        for card in cards:
            key.append(int(card.name))
        key.sort()
        actions.append(actions_lookuptable[str(key)])

    # yaobuqi
    if len(actions) == 0:
        actions.append(430)
    # buyao
    elif game.last_move != "start":
        actions.append(429)

    return actions


# 结合state和可以出的actions作为新的state
def combine(s, a):
    for i in a:
        s[33 + i] = 1
    return s


############################################
#                 random                    #
############################################
def choose_random(next_move_types, next_moves, last_move_type):
    # 要不起
    if len(next_moves) == 0:
        return "yaobuqi", []
    else:
        # start不能不要
        if last_move_type == "start":
            r_max = len(next_moves)
        else:
            r_max = len(next_moves) + 1
        r = np.random.randint(0, r_max)
        # 添加不要
        if r == len(next_moves):
            return "buyao", []

    return next_move_types[r], next_moves[r]

# def config():
#     self.actions_lookuptable = action_dict
#     self.dim_actions = len(self.actions_lookuptable) + 2  # 429 buyao, 430 yaobuqi
#     self.dim_states = 30 + 3 + 431  # 431为dim_actions

if __name__ == "__main__":
    g = Poker()
    g.game_start(True)
    done = False
    while (not done):
        old_state = get_state(g.playrecords, g.players[0])
        print(" -----------------------")
        print(old_state)
        print(" -----------------------")
        # next_move_types, next_moves = g.get_next_moves()
        # actions = get_actions(next_moves, g)
        winner, done = g.get_next_move(action="mcts")
        new_state = get_state(g.playrecords, g.players[0])

    print(" -----------------------")
    print(new_state)
    print(winner)
