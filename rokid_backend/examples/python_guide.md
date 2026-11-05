# Python data types

## Mutability

Lists, dictionaries and sets are mutable containers: their contents can change after creation.
A tuple is an immutable sequence. Once a tuple is created its elements cannot be added, removed or replaced.
Strings are also immutable: every operation that seems to modify a string returns a new string.

## Comprehensions

A list comprehension builds a list from an iterable in a single expression, for example [x * 2 for x in range(3)].
A generator expression uses parentheses and produces items lazily, which saves memory for large sequences.

# SQL basics

## Filtering

The WHERE clause filters rows before aggregation. The HAVING clause filters groups after GROUP BY has been applied.
ORDER BY sorts the result set and LIMIT restricts the number of returned rows.

## Joins

An INNER JOIN returns only rows with matching keys in both tables. A LEFT JOIN keeps every row of the left table and fills missing right-side columns with NULL.
