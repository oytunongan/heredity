import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):

    jp = 1
    for person in people:
        if check_parents(people, person):
            gene = calc_prob(people, person, one_gene, two_genes)
            if person in have_trait:
                jp *= list(gene.values())[0] * PROBS["trait"][list(gene.keys())[0]][True]
            else:
                jp *= list(gene.values())[0] * PROBS["trait"][list(gene.keys())[0]][False]
        else:
            if person in one_gene and person in have_trait:
                jp *= PROBS["gene"][1] * PROBS["trait"][1][True]
            elif person in two_genes and person in have_trait:
                jp *= PROBS["gene"][2] * PROBS["trait"][2][True]
            elif person in one_gene and person not in have_trait:
                jp *= PROBS["gene"][1] * PROBS["trait"][1][False]
            elif person in two_genes and person not in have_trait:
                jp *= PROBS["gene"][2] * PROBS["trait"][2][False]
            elif person not in one_gene and person not in two_genes and person in have_trait:
                jp *= PROBS["gene"][0] * PROBS["trait"][0][True]
            elif person not in one_gene and person not in two_genes and person not in have_trait:
                jp *= PROBS["gene"][0] * PROBS["trait"][0][False]
    return jp

    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    # raise NotImplementedError


def update(probabilities, one_gene, two_genes, have_trait, p):
    for person in probabilities:
        if person in one_gene:
            probabilities[person]["gene"][1] += p
        elif person in two_genes:
            probabilities[person]["gene"][2] += p
        else:
            probabilities[person]["gene"][0] += p
        if person in have_trait:
            probabilities[person]["trait"][True] += p
        else:
            probabilities[person]["trait"][False] += p

    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    # raise NotImplementedError


def normalize(probabilities):
    for person in probabilities:
        sum_genes = 0
        sum_traits = 0
        for i in range(3):
            sum_genes += probabilities[person]["gene"][i]
        sum_traits = probabilities[person]["trait"][True] + probabilities[person]["trait"][False]
        if sum_genes != 0:
            coef_genes = 1 / sum_genes
        else:
            coef_genes = 0
        if sum_traits != 0:
            coef_traits = 1 / sum_traits
        else:
            coef_traits = 0
        for i in range(3):
            probabilities[person]["gene"][i] *= coef_genes
        probabilities[person]["trait"][True] *= coef_traits
        probabilities[person]["trait"][False] *= coef_traits

    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    # raise NotImplementedError


def check_parents(people, person):
    if people[person]['mother'] and people[person]['father']:
        return True


def calc_prob(people, person, one, two):
    probability = {}
    m = people[person]['mother']
    f = people[person]['father']
    if person in one:
        if m not in one and m not in two:
            if f not in one and f not in two:
                probability[1] = 2 * PROBS["mutation"] * (1 - PROBS["mutation"])
            elif f in two:
                probability[1] = PROBS["mutation"] * PROBS["mutation"] + \
                    (1 - PROBS["mutation"]) * (1 - PROBS["mutation"])
            elif f in one:
                probability[1] = 0.5 * (1 - PROBS["mutation"]) + 0.5 * PROBS["mutation"]
        elif m in two:
            if f not in one and f not in two:
                probability[1] = PROBS["mutation"] * PROBS["mutation"] + \
                    (1 - PROBS["mutation"]) * (1 - PROBS["mutation"])
            elif f in two:
                probability[1] = 2 * PROBS["mutation"] * (1 - PROBS["mutation"])
            elif f in one:
                probability[1] = 0.5 * (1 - PROBS["mutation"]) + 0.5 * PROBS["mutation"]
        elif m in one:
            if f not in one and f not in two:
                probability[1] = 0.5 * (1 - PROBS["mutation"]) + 0.5 * PROBS["mutation"]
            elif f in two:
                probability[1] = 0.5 * (1 - PROBS["mutation"]) + 0.5 * PROBS["mutation"]
            elif f in one:
                probability[1] = 2 * 0.5 * 0.5
    elif person in two:
        if m not in one and m not in two:
            if f not in one and f not in two:
                probability[2] = PROBS["mutation"] * PROBS["mutation"]
            elif f in two:
                probability[2] = PROBS["mutation"] * (1 - PROBS["mutation"])
            elif f in one:
                probability[2] = 0.5 * PROBS["mutation"]
        elif m in two:
            if f not in one and f not in two:
                probability[2] = PROBS["mutation"] * (1 - PROBS["mutation"])
            elif f in two:
                probability[2] = (1 - PROBS["mutation"]) * (1 - PROBS["mutation"])
            elif f in one:
                probability[2] = 0.5 * (1 - PROBS["mutation"])
        elif m in one:
            if f not in one and f not in two:
                probability[2] = 0.5 * PROBS["mutation"]
            elif f in two:
                probability[2] = 0.5 * (1 - PROBS["mutation"])
            elif f in one:
                probability[2] = 0.5 * 0.5
    else:
        if m not in one and m not in two:
            if f not in one and f not in two:
                probability[0] = (1 - PROBS["mutation"]) * (1 - PROBS["mutation"])
            elif f in two:
                probability[0] = PROBS["mutation"] * (1 - PROBS["mutation"])
            elif f in one:
                probability[0] = 0.5 * (1 - PROBS["mutation"])
        elif m in two:
            if f not in one and f not in two:
                probability[0] = PROBS["mutation"] * (1 - PROBS["mutation"])
            elif f in two:
                probability[0] = (1 - PROBS["mutation"]) * (1 - PROBS["mutation"])
            elif f in one:
                probability[0] = 0.5 * PROBS["mutation"]
        elif m in one:
            if f not in one and f not in two:
                probability[0] = 0.5 * (1 - PROBS["mutation"])
            elif f in two:
                probability[0] = 0.5 * PROBS["mutation"]
            elif f in one:
                probability[0] = 0.5 * 0.5
    return probability


if __name__ == "__main__":
    main()
